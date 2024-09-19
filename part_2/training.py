import sys
import math
import argparse
import logging
import numpy as np
import datetime
import torch
from torch import nn, optim
from dataset import LunaDataset
from torch.utils.data import DataLoader


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias = True)

        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(in_channels, conv_channels, kernel_size = 3, padding = 1, bias = True)

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

METRICS_LABEL_NDX = 0    # These are the named array indices which are declared at the module level.
METRICS_PRED_NDX = 1     # These are the named array indices which are declared at the module level.
METRICS_LOSS_NDX = 2    # These are the named array indices which are declared at the module level.
METRICS_SIZE = 3        # These are the named array indices which are declared at the module level.

def enumerateWithEstimate(
        iter,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,
):
    """
    In terms of behavior, `enumerateWithEstimate` is almost identical
    to the standard `enumerate` (the differences are things like how
    our function returns a generator, while `enumerate` returns a
    specialized `<enumerate object at 0x...>`).
    However, the side effects (logging, specifically) are what make the
    function interesting.
    :param iter: `iter` is the iterable that will be passed into
        `enumerate`. Required.
    :param desc_str: This is a human-readable string that describes
        what the loop is doing. The value is arbitrary, but should be
        kept reasonably short. Things like `"epoch 4 training"` or
        `"deleting temp files"` or similar would all make sense.
    :param start_ndx: This parameter defines how many iterations of the
        loop should be skipped before timing actually starts. Skipping
        a few iterations can be useful if there are startup costs like
        caching that are only paid early on, resulting in a skewed
        average when those early iterations dominate the average time
        per iteration.
        NOTE: Using `start_ndx` to skip some iterations makes the time
        spent performing those iterations not be included in the
        displayed duration. Please account for this if you use the
        displayed duration for anything formal.
        This parameter defaults to `0`.
    :param print_ndx: determines which loop interation that the timing
        logging will start on. The intent is that we don't start
        logging until we've given the loop a few iterations to let the
        average time-per-iteration a chance to stablize a bit. We
        require that `print_ndx` not be less than `start_ndx` times
        `backoff`, since `start_ndx` greater than `0` implies that the
        early N iterations are unstable from a timing perspective.
        `print_ndx` defaults to `4`.
    :param backoff: This is used to how many iterations to skip before
        logging again. Frequent logging is less interesting later on,
        so by default we double the gap between logging messages each
        time after the first.
        `backoff` defaults to `2` unless iter_len is > 1000, in which
        case it defaults to `4`.
    :param iter_len: Since we need to know the number of items to
        estimate when the loop will finish, that can be provided by
        passing in a value for `iter_len`. If a value isn't provided,
        then it will be set by using the value of `len(iter)`.
    :return:
    """
    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    logging.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    start_ts = datetime.time()
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            # ... <1>
            duration_sec = ((datetime.time() - start_ts)
                            / (current_ndx - start_ndx + 1)
                            * (iter_len-start_ndx)
                            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            logging.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = datetime.time()

    logging.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))

class LunaTrainingApp():

    def __init__(self, sys_argv = None):
        # Here we are checking of arguments are provided by the user in the CLI. If not then we use default system arguments
        if sys_argv == None: # check 
            sys_argv = sys.argv[1:]  # using the defualt system arguments

        # Then we instantiate the argument parser object.
        parser = argparse.ArgumentParser()

        # Now we add an argument --num-workers which lets us specify how many backgroung workers would be utilized for data loading.
        parser.add_argument('--num-workers',
                                help = 'number of worker processes for background data loading',
                                type = int,
                                default = 8)
        parser.add_argument("--batch-size",
                                help = 'The size of each batch of images',
                                type = int,
                                default = 32)
        parser.add_argument("--epochs",
                                help = 'number of iteration of training on the full dataset',
                                type = int,
                                default = 1)

        # Then we parse arguments provided in the CLI and assign them to an attribute cli_args
        self.cli_args = parser.parse_args(sys_argv)

        # We instantiate a datetime.now object and assign it to the time_str attribute.
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        # Now we will initialize the model and the optimizer

        # First we will check if the GPU is available
        self.use_cuda = torch.cuda.is_available()
        # If GPU is available then use GPU else use CPU
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        # Initialize the model 
        self.model = self.initModel()

        #Initialize the optimizer
        self.optimizer = self.initOptimizer()

    def initModel(self):
        """ This function initialzes the model and transfers the model and parameters to the GPU.
            If multiple GPU's are available then execute the model computations in paraller and sync and return the results """

        # initialize the model
        model = LunaModel()

        
        if self.use_cuda:
            logging.info(f"Using CUDA : {torch.cuda.device_count()} devices")
            # If multiple GPU's are available then execute the computations in parallel
            if torch.cuda.device_count()>1:
                model = nn.DataParallel(model)
            # move the model to the device
            model = model.to(self.device)
        return model
    
    def initOptimizer(self):
        return optim.SGD(self.model.parameters(), lr= 0.001,momentum = 0.99)

    # Now lets put the training data into the dataloader.
    def initTrainDL(self):
        train_ds = LunaDataset(val_stride= 10, is_val_set_bool = False)


        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(train_ds,
                                batch_size = batch_size,
                                num_workers = self.cli_args.num_workers,
                                pin_memory = self.use_cuda)
        return train_dl

    def initValDl(self):
        val_ds = LunaDataset(val_stride = 10, is_val_set_bool=True)
        batch_size = self.cli_args.batch_size

        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        
        val_dl = DataLoader(val_ds,
                            batch_size = batch_size,
                            num_workers= self.cli_args.num_workers,
                            pin_memory = self.use_cuda)

        return val_dl
    
    def logMetrics(self, epochs_ndx, mode_str, metrics_t, classification_threshold):
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classification_threshold   # Non Nodules
        negPred_mask = metrics_t[METRICS_LABEL_NDX] >= classification_threshold    # Nodules


        posLabel_mask = ~negLabel_mask
        posPred_mask = ~posLabel_mask


        # Next we  would use the masks to calculate some per label statistics and use them to store in a dictionary metrics_dict
        metrics_dict = {}
        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()
        metrics_dict['correct/all'] = metrics_t(pos_correct + neg_correct)/ np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct/ np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct/ np.float32(pos_count) * 100


        logging.info(f"E{epochs_ndx}, {mode_str[:8]}, {metrics_dict['loss/all']:.4f} LOSS, {metrics_dict['correct/all']:-5.1f}% CORRECT")
        logging.info(f"E{epochs_ndx}, {mode_str[:8]}, {metrics_dict['loss/neg']:.4f} LOSS, {metrics_dict['correct/neg']:-5.1f}% CORRECT {neg_correct} of {neg_count}")
        logging.info(f"E{epochs_ndx}, {mode_str[:8]}, {metrics_dict['loss/pos']:.4f} LOSS, {metrics_dict['correct/pos']:-5.1f}% CORRECT {pos_correct} of {pos_count}")





    

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size,  metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking = True)
        label_g = label_t.to(self.device, non_blocking = True)


        logits_g, probability_g = self.model(input_g)
        loss_fn = nn.CrossEntropyLoss(reduction = 'none')  # reduction = 'none' gives the loss per sample

        loss_g = loss_fn(logits_g, label_g[:,1])

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:,1].detach()            # We use detach since none of them need to hold gradients.
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g[:,1].detach()


        return loss_g.mean()  # recombines the loss per sample to a single value averaged over the entire batch 


    def doTraining(self, epochs_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device = self.device)  # Initializes an empty array

        batch_iter = enumerateWithEstimate(train_dl, f"E{epochs_ndx} Training",  # Sets up batch looping with time estimate
                                            start_ndx = train_dl.num_workers)

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()   # Frees up any leftover gradient tensors

            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)

            loss_var.backward()     # Backpropagates

            self.optimizer.step()   # Updates the model weights

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')
            

    def doValidation(self, epochs_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()

            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device = self.device)

            batch_iter = enumerateWithEstimate(val_dl, f"E{epochs_ndx} Validation", start_ndx = val_dl.num_workers)
            
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')   # We would need to measure the validation statistics for each sample


    def main(self):
        
        logging.info(f"Starting {type(self).__name__}, {self.cli_args}")
        train_dl = self.initTrainDL()
        val_dl = self.initValDl()

        for epochs_ndx in range(1, self.cli_args.epochs + 1):
            trnMetrics_t = self.doTraining(epochs_ndx, train_dl)
            self.logMetrics(epochs_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epochs_ndx, val_dl)
            self.logMetrics(epochs_ndx, 'val', valMetrics_t)

if __name__ == '__main__':
    LunaTrainingApp().main()