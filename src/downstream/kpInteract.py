import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300

import wandb

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class RelationModel(nn.Module):
    # A model that takes all information about the relations in the graph 
    # and outputs effects of all interactions between objects
    def __init__(self, input_size, output_size, hidden_size, num_relations, batch_norm):
        super().__init__()
        I=input_size
        O=output_size
        H=hidden_size
        R=num_relations
        inputs= [I, H, H, H]
        outputs= inputs[1:]+[O]
        layers=[]
        for i in range(len(inputs)):
            layers.append(nn.Linear(inputs[i], outputs[i]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(R))
            layers.append(nn.ReLU())
        self.model=nn.Sequential(*layers)

    def forward(self, x):
        x=self.model(x)
        return x

class ObjectModel(nn.Module):
    # A model takes information about all objects and effects on them,
    # and outputs prediction of the next state of the graph
    def __init__(self, input_size, hidden_size, num_objects, batch_norm):
        super().__init__()
        I=input_size
        H=hidden_size
        R=num_objects
        inputs= [I, H, H, H]
        outputs= inputs[1:]+[I]
        layers=[]
        for i in range(len(inputs)):
            layers.append(nn.Linear(inputs[i], outputs[i]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(R))
            layers.append(nn.ReLU())
        self.model=nn.Sequential(*layers)

    def forward(self, x):
        x=self.model(x)
        return x

class ProjectionLayer(nn.Module):
    # A model takes information about all objects and effects on them,
    # and outputs prediction of the next state of the graph
    def __init__(self, input_size, output_size, num_keypoints, batch_norm):
        super().__init__()
        I=input_size
        O=output_size
        KP=num_keypoints
        inputs= [I]
        outputs= inputs[1:]+[O]
        layers=[]
        for i in range(len(inputs)):
            layers.append(nn.Linear(inputs[i], outputs[i]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(KP))
            layers.append(nn.ReLU())
        self.model=nn.Sequential(*layers)

    def forward(self, x):
        x=self.model(x)
        return x

class InteractionNetwork(nn.Module):
    # The whole interaction network model, predicts the next state of the graph
    def __init__(self, num_objects, object_dim, num_relations, relation_dim, effect_dim, batch_norm):
        super().__init__()
        self.relational_model=RelationModel(
            input_size= 2*object_dim + relation_dim,
            output_size= effect_dim,
            hidden_size=100,
            num_relations=num_relations,
            batch_norm=batch_norm
        )
        self.object_model=ObjectModel(
            input_size= object_dim + effect_dim,
            hidden_size=100,
            num_objects=num_objects,
            batch_norm=batch_norm
        )

    def forward(self, objects, sender_relations, receiver_relations, relation_info):
        senders = sender_relations.permute(0,2,1)
        senders = senders @ objects
        receivers = receiver_relations.permute(0,2,1)
        receivers = receivers @ objects
        relational_inputs = torch.cat([senders, receivers, relation_info], dim=-1)
        effects = self.relational_model(relational_inputs)
        effect_receivers = receiver_relations @ effects
        object_inputs = torch.cat([objects, effect_receivers],dim=-1)
        predicted_objects = self.object_model(object_inputs)
        return predicted_objects

class KeypointInteractionNetwork(nn.Module):
    def __init__(self, config, task, action_dim=None):
        super().__init__()
        self.interaction_layers=config['interaction_layers']
        self.latent_size=config['interaction_latent_size']
        self.history=config['interaction_history']
        self.horizon=config['prediciton_horizon']
        self.batch_norm=config['batch_norm']
        KP = config['num_keypoints']
        keypoint_dim=3 # x, y, activation_status
        self.encoder=nn.Sequential(
            nn.Linear(self.history*keypoint_dim+1, self.latent_size), # +1 for the positional encoding
            nn.ReLU(),
            # nn.BatchNorm1d(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
        )
        # number of objects is 2 * the number of keypoints (to time steps)
        self.num_objects = KP
        # the features of each node here is the latent size
        self.object_dim = self.latent_size
        # number of relation is the number of edges in a fully connected graph
        self.num_relations= KP * (KP - 1)
        self.relation_dim= 1
        self.effect_dim=100
        self.interaction_network=InteractionNetwork(
            num_objects = self.num_objects,
            object_dim = self.object_dim,
            num_relations=self.num_relations,
            relation_dim= self.relation_dim,
            effect_dim=self.effect_dim,
            batch_norm=self.batch_norm
        )
        # a projection layer for multistep interactions
        # self.projection_layer=ProjectionLayer(self.object_dim+self.effect_dim, self.object_dim, KP, self.batch_norm)
        self.projection_layer=nn.Linear(self.object_dim+self.effect_dim, self.object_dim)
        # receiver relation and sender relations are one hot encoding
        # each column indicates the receiver and sender object's index
        self.receiver_relations=torch.zeros((1,self.num_objects, self.num_relations)).to(device)
        self.sender_relations=torch.zeros((1,self.num_objects, self.num_relations)).to(device)
        r=0 # relation index
        # create a relation between each two object
        for i in range(self.num_objects):
            for j in range(self.num_objects):
                if (i!=j):
                    # The relation at index r has object i as receiver and object j as sender
                    self.receiver_relations[0,i,r] = 1.0
                    self.sender_relations[0,j,r] = 1.0
                    r+=1
        # There is no relation infos for keypoints
        self.relation_info=torch.zeros((1,self.num_relations, self.relation_dim)).to(device)
        # positional encoder a number between -1 and 1 for each keypoint
        self.positional_encoding=torch.linspace(-1,1,KP).to(device)[None,:,None]
        # decode the keypoints
        self.decoder=nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.horizon*keypoint_dim),
        )
        self.task=task
        if task=='imitation':
            # magical gives 4 images in the demonstration, we follow that
            I=self.latent_size*KP*4
            H=self.latent_size
            O=action_dim
            inputs= [I, 4*H, 2*H, H]
            outputs= inputs[1:]+[H]
            layers=[]
            # I2=self.latent_size*KP
            # self.l1=nn.Linear(I2, H)
            # self.rnn=nn.RNN(input_size=H, hidden_size=H, num_layers = 2, batch_first=True)
            # self.l2=nn.Linear(H, O)
            # self.hidden=torch.zeros(2,config['batch_size'],self.latent_size).to(device)
            for i in range(len(inputs)):
                layers.append(nn.Linear(inputs[i], outputs[i]))
                if self.batch_norm:
                    layers.append(nn.BatchNorm1d(outputs[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(H, O))
            self.action_model=nn.Sequential(*layers)
            self.action_model=nn.Sequential(
                nn.Linear(I, self.latent_size),
                nn.ReLU(),
                nn.Linear(self.latent_size, O),
            )

    def interaction_step(self, x):
        # pass the keypoints to the interaction network
        N,KP,L=x.shape
        objects=x
        sender_relations=self.sender_relations.repeat(N,1,1)
        receiver_relations=self.receiver_relations.repeat(N,1,1)
        relation_info=self.relation_info.repeat(N,1,1)
        # new embeding after one pass throught the interaction network
        output=self.interaction_network(objects, sender_relations, receiver_relations, relation_info)
        return output
    
    def post_process_dynamics(self, x, N, SF, KP):
        # flatten the keypoints over batches
        x=x.view(-1,self.latent_size)
        # pass through the decoder to get predicitons
        x=self.decoder(x)
        # redestribute the keypoints
        x=x.view(N,SF,KP,self.horizon,-1)
        # pass the coordinates through a tanh to keep them between [-1,1]
        x[...,0]=torch.tanh(x[...,0])
        x[...,1]=torch.tanh(x[...,1])
        return x
    
    def post_process_imitation(self, x, N, SF, KP):
        # flatten the keypoints embedding in each batch
        # x=x.view(N, SF, -1)
        # h=F.relu(self.l1(x))
        # a, self.hidden=self.rnn(h, self.hidden)
        # x=self.l2(a)[:,-1]
        x=x.view(N, -1)
        x=self.action_model(x)
        # x.register_hook(lambda grad: print("grad : ",grad.mean()))
        return x

    def forward(self, keypoints):
        N,SF,KP,F=keypoints.shape
        shifted_keypoints=keypoints.clone()
        for i in range(self.history-1):
            # shift the keypoints over time
            shifted_keypoints=torch.roll(shifted_keypoints,1,dims=1)
            # concatenate the keypoints and the shifted one
            keypoints= torch.cat([shifted_keypoints, keypoints], dim=-1)
        # flatten the keypoints over batches
        positional_encoding=self.positional_encoding.repeat(N,SF,1,1)
        keypoints=torch.cat((positional_encoding, keypoints), dim=-1)
        x=keypoints.view(N*SF,KP,-1)
        x=self.encoder(x)
        for i in range(self.interaction_layers):
            x=self.interaction_step(x)
            x=self.projection_layer(x)
        predictions=self.post_process_dynamics(x, N, SF, KP)
        if self.task=='imitation':
            actions=self.post_process_imitation(x, N, SF, KP)
            return predictions, actions
        else:
            return predictions