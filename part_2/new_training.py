from dataset import LunaDataset
import sys
import argparse
import datetime
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import math
from tqdm import tqdm



class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias = True)

        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size = 3, padding = 1, bias = True)

        self.relu2 = nn.ReLU(inplace = True)

        self.max_pool = nn.MaxPool3d(2,2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)     # this could be implemented as calls to functional API instead
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)     # this could be implemented as calls to functional API instead
        block_out = self.max_pool(block_out)  # this could be implemented as calls to functional API instead

        return block_out


class LunaModel(nn.Module):
    def __init__(self, in_channels = 1 , conv_channels = 8):
        super().__init__()
        
        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels*2)
        self.block3 = LunaBlock(conv_channels*2, conv_channels*4)
        self.block4 = LunaBlock(conv_channels*4, conv_channels*8)

        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)
    
    def _init_weights(self):
        for m in self.modules:
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_out', nonlinearity='relu')
            if m.bias is not None:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / math.sqrt(fan_out)
                nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):

        bn_output = self.tail_batchnorm(input_batch)

        block_1_out = self.block1(bn_output)
        block_2_out = self.block2(block_1_out)
        block_3_out = self.block3(block_2_out)
        block_4_out = self.block4(block_3_out)

        conv_flat = block_4_out.view(block_4_out.size(0), -1)  # Flattening to (batch_size, -1)

        linear_output = self.head_linear(conv_flat)
        softmax_output = self.head_softmax(linear_output)

        return linear_output, softmax_output

# We first need to create the class of our training app and create a main function
class TrainingApp():
    """ This class defines the training app with all the functionality required for the app to work"""
    def __init__(self, sys_argv = None):
        """This function initializes the app object, accepts all the passed in parameters from the CLI and assigns them to attributes"""
        if sys_argv == None: # If no system arguments are provided then use the default ones
            sys_argv = sys.argv[1:]
        
        # Define the parser with a description of what it does
        parser = argparse.ArgumentParser(description="Parsing command line arguments for Lung Cancer Detection Script")

        # These are the arguments that the parser can accept from the user. 
        parser.add_argument('--epochs',
                            help='How many epochs training should run',
                            type= int,
                            default = 1)

        parser.add_argument('--num-workers',
                            help='Number of background worker processes for loading data',
                            type = int,
                            default = 4)
        
        parser.add_argument('--batch-size',
                            help = 'Provide the batch size of images',
                            type = int,
                            default = 2)

        self.cli_args = parser.parse_args(sys_argv)

        # print("printing CLI args")
        # print(self.cli_args)
        # print(self.cli_args.batch_size)


        # We instantiate a datetime.now object and assign it to the time_str attribute.
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        # Check if a GPU is available for use
        self.use_cuda = torch.cuda.is_available()

        # If GPU is available then use the GPU as the device else use the CPU as the device.
        self.device = torch.device("cuda" if self.use_cuda else 'cpu')

        # Now initialize the model object and assign it to the self.model attribute
        self.model = self.initModel()

        # Now initialize the optimizer
        self.optimizer = self.initOptimizer()
        

    def initModel(self):
        """ This function intializes the model that will be used for the classification task."""

        # create a model object
        model = LunaModel()

        # We can also make the model use multiple GPU's if multiple GPU's are available
        # First check if a GPU is available
        if self.use_cuda:
            # Logging capcbilities to be addded later here.
            if torch.cuda.device_count() > 1:
                model = nn.Dataparallel(model)  # We can have the model initialized parallely on multiple GPU's
            
            # Transport the model parameters to GPU
            model = model.to(self.device)

        return model

    def initOptimizer(self):
        return optim.SGD(self.model.parameters(), lr = 0.001, momentum = 0.99)
    
    # Now we have defined the model and the Optimizer and also accepted the arguments from the user.
    # Now we to get our training and validation datasets and set up a dataloader to batch and load the data

    def initTrainDL(self):
        # Import the dataset
        train_ds = LunaDataset(val_stride = 10, is_val_set_bool=False)
        # print(type(train_ds))
        # # print(train_ds[0])
        # print(len(train_ds[0]))
        # # print(train_ds.shape)


        # Get the batch_size from the user input which is stored in the cli_args list
        batch_size = self.cli_args.batch_size
                
        if self.use_cuda:
            # If GPU is avaibale then scale the batch_size by the number of GPU's because you will do parallel processing if we have multiple GPU's
            batch_size *= torch.cuda.device_count()

        train_DL = DataLoader(batch_size=batch_size,
                              dataset= train_ds,
                              num_workers= self.cli_args.num_workers,
                              pin_memory=self.use_cuda)
        # for batch in train_DL:
        #     print(len(batch))
        return train_DL
    
    def initValDL(self):
        val_ds = LunaDataset(val_stride = 10, is_val_set_bool=True)

        batch_size = self.cli_args.batch_size

        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        
        val_DL = DataLoader(batch_size=batch_size,
                            dataset=val_ds,
                            num_workers = self.cli_args.num_workers,
                            pin_memory= self.use_cuda)
        
        return val_DL
    
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size):
        # From each batch tuple take out the input and the label and other information
        input_t, label_t, _series_uid, _center_list = batch_tup

        # Move the inputs and the labels to the GPU or the cpu whichever device you machine has
        input_g = input_t.to(self.device, non_blocking = True)
        label_g = label_t.to(self.device, non_blocking = True)
   
        # now run the model and get the logits and the probabilities
        logits_g, probability_g  = self.model(input_g)
        # calculate the loss using the loss_fn
        loss_fn = nn.CrossEntropyLoss(reduction='none')  # The reduction parameter returns losses for individual samples

        loss_g = loss_fn(logits_g, label_g[:, 1])

        # We can also store all the pred, probility and loss values for individual samples in a dict for displaying purposes. 
        # This would be added later
        return loss_g.mean()

    # Now the only thing that is left to do is to define the training loop
    def doTraining(self, train_DL, epochs_ndx):
        # Tells pytorch that the model is training
        self.model.train()

        batch_iter = tqdm(enumerate(train_DL))

        loss_per_epoch = 0

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, self.cli_args.batch_size)

            loss_per_epoch += loss_var

            loss_var.backward()

            self.optimizer.step()
        return loss_per_epoch


    def main(self):
        """ This function will be the first function that will run when the app is running.
            It will be responsible for performing the training loop of the app."""

        train_dl = self.initTrainDL()
        for epoch in range(self.cli_args.epochs):
            loss_per_epoch = self.doTraining(train_dl, epoch)
            print(f"The Loss Per Epoch is {loss_per_epoch}")


# Now the first thing that we will do is that we would build the main function of our app which would instantiate the Training_app object and run it
if  __name__ == "__main__":
    TrainingApp().main()
    # print(sys.argv)
    # print(x.__init__)
