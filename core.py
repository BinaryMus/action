import torch
import time

from copy import deepcopy

class FederatedLearning:
    def __init__(self, global_model, client_models, criterions, optimizers, schedulers, dataloaders, valloader, device, comp, global_epoch, local_epoch=1, beta=0.9, estimate_type='last', nlp=False, **kwargs):
        self.global_model = global_model
        self.client_models = client_models
        self.num_client = len(client_models)
        self.client_models = client_models

        self.names = self.global_model.state_dict().keys()
        self.tilde_gradients = {name: torch.zeros_like(param).to(device) for name, param in self.global_model.state_dict().items()}

        self.criterions = criterions
        self.dataloaders = dataloaders
        self.optimizers = optimizers
        self.schedulers = schedulers

        self.valloader = valloader
        self.comps = [deepcopy(comp) for _ in range(self.num_client)]
        # self.comp = comp

        self.beta = beta
        self.estimate_type = estimate_type
        self.global_epoch = global_epoch
        self.local_epoch = local_epoch
        self.device = device
        self.nlp = nlp
        self.kwargs = kwargs
 
    def run(self):
        loss_lst = []
        acc1_lst = []
        acc5_lst = []
        for epoch in range(self.global_epoch):
            start_time = time.time()
            all_client_gradients = {}
            avg_loss_lst = []
            avg_error_lst = []

            for client_idx in range(self.num_client):
                gradient, avg_loss = self.train(client_idx)
                avg_loss_lst.append(avg_loss)
                gradient_recover, error = self.compression(client_idx, gradient)
                avg_error_lst.append(error)
                for i, v in gradient_recover.items():
                    if i in all_client_gradients:
                        all_client_gradients[i].append(v)
                    else:
                        all_client_gradients[i] = [v]
            
            self.aggregate(all_client_gradients)
            loss_avg = sum(avg_loss_lst) / len(avg_loss_lst)
            error_avg = round(sum(avg_error_lst) / len(avg_error_lst), 2)
            acc1, acc5 = self.validate()

            loss_lst.append(loss_avg)
            acc1_lst.append(acc1)
            acc5_lst.append(acc5)

            loss_avg = round(loss_avg, 4)
            acc1 = round(acc1, 4)
            acc5 = round(acc5, 4)
            t = round(time.time() - start_time, 1)
            print(f'{epoch=}  {error_avg=}  {loss_avg=}  {acc1=} {acc5=} {t=}')


        return loss_lst, acc1_lst, acc5_lst

    def train(self, client_idx):
        client = self.client_models[client_idx]
        dataloader = self.dataloaders[client_idx]
        criterion = self.criterions[client_idx]
        scheduler = self.schedulers[client_idx]
        optimizer = self.optimizers[client_idx]
        client.train()
        start_time = time.time()

        initial_weights = deepcopy(client.state_dict())

        for epoch in range(self.local_epoch):
            running_loss = 0.0
            if not self.nlp:
                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = client(data)
                    loss = criterion(output, target)
                    loss.backward()
                    running_loss += loss.item()
                    optimizer.step()
                scheduler.step()
            else:
                for batch in dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    labels = batch["labels"]

                    optimizer.zero_grad()
                    output = client(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    loss = criterion(output.logits, labels)
                    loss.backward()
                    running_loss += loss.item()
                    optimizer.step()
            # print(f"Client@{client_idx}: Local epoch={epoch} loss={running_loss / len(dataloader)} time={time.time()-start_time}")

        gradient = {name: client.state_dict()[name] - initial_weights[name] for name in self.names}

        del initial_weights
        return gradient, running_loss / len(dataloader)
    
    def validate(self):
        with torch.no_grad():
            self.global_model.eval()
            total, correct1, correct5 = 0, 0, 0
            if not self.nlp:
                for data, target in self.valloader:
                    data, target = data.to(self.device), target.to(self.device)
                    total += len(data)
                    output = self.global_model(data)
                    predict = output.argmax(dim=1)
                    correct1 += torch.eq(predict, target).sum().float().item()
                    target_resize = target.view(-1, 1)
                    _, predict = output.topk(5)
                    correct5 += torch.eq(predict, target_resize).sum().float().item()
            else:
                for batch in self.valloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    target = batch["labels"]
                    total += batch["input_ids"].size(0)

                    output = self.global_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    predict = output.logits.argmax(dim=1)
                    correct1 += torch.eq(predict, target).sum().float().item()
                    target_resize = target.view(-1, 1)
                    _, predict_top5 = output.logits.topk(5, dim=1)
                    correct5 += torch.sum(predict_top5 == target_resize).float().item()
            acc1 = correct1 / total
            acc5 = correct5 / total
            return acc1, acc5

    def compression(self, client_idx, gradient):
        with torch.no_grad():
            comp = self.comps[client_idx]
            # recovered_gradient = {name: self.comp.decode(*self.comp.encode(gradient[name].flatten())).view(gradient[name].shape) for name in self.names}
            grad_info = {
                name: (grad.shape, grad.numel()) 
                for name, grad in gradient.items()
            }
            all_grads = [g.flatten() for g in gradient.values()]
            all_grads = torch.cat([g.flatten() for g in gradient.values()])
            all_tilde_gradients = torch.cat([g.flatten() for g in self.tilde_gradients.values()])
            all_grads_recover = comp.decode(*comp.encode(all_grads, tilde_gradients=all_tilde_gradients))
            error = (all_grads_recover - all_grads).norm(2).item()
            # print(f'Error: {(all_grads_recover - all_grads).norm(2).item()}')
            recovered_gradient = {}
            pointer = 0
            for name, (shape, numel) in grad_info.items():
                recovered_gradient[name] = all_grads_recover[pointer:pointer + numel].reshape(shape)
                pointer += numel
        return recovered_gradient, error
    
    def aggregate(self, gradients):
        with torch.no_grad():
            aggregate_gradient = {}
            aggregate_param = {}
            for name, param in self.global_model.state_dict().items():
                aggregate_gradient[name] = sum(gradients[name]) / self.num_client
                if self.estimate_type == 'last':
                    self.tilde_gradients[name] = aggregate_gradient[name]
                elif self.estimate_type == 'avg':
                    self.tilde_gradients[name] = self.beta * self.tilde_gradients[name] + (1 - self.beta) * aggregate_gradient[name]
                aggregate_param[name] = param.to(self.device) + aggregate_gradient[name]
            self.global_model.load_state_dict(aggregate_param)
            for i in range(self.num_client):
                self.client_models[i].load_state_dict(aggregate_param)



