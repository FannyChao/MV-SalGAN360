import theano.tensor as T


class Model(object):
    def __init__(self, input_width, input_height, batch_size=32):
        
        self.inputWidth = input_width
        self.inputHeight = input_height
        
        self.inputWidth_C = 512
        self.inputHeight_C = 256
 
        self.G_lr = None
        self.D_lr = None

        self.momentum = None

        self.net = None
        self.net90 = None
        self.net120 = None
        self.net180 = None
        self.discriminator = None
        self.batch_size = batch_size

        self.D_trainFunction = None
        self.D_trainFunction90 = None
        self.D_trainFunction120 = None
        self.D_trainFunction180 = None
        self.D_trainFunction_C = None
        self.G_trainFunction = None
        self.G_trainFunction90 = None
        self.G_trainFunction120 = None
        self.G_trainFunction180 = None
        self.G_trainFunction_C = None
        self.predictFunction = None
        self.predictFunction90 = None
        self.predictFunction120 = None
        self.predictFunction180 = None
        
        self.input_var = T.tensor4()
        self.input_var90 = T.tensor4()
        self.input_var120 = T.tensor4()
        self.input_var180 = T.tensor4() 
        self.input_var_C = T.tensor4()
        self.input_var_C_im = T.tensor4()
        
        self.output_var = T.tensor4()
        self.output_var_sal = T.tensor4()
        self.output_var_sal_C = T.tensor4()
        self.output_var_sal90 = T.tensor4()
        self.output_var_sal120 = T.tensor4()
        self.output_var_sal180 = T.tensor4()
        
        self.output_var_fixa = T.tensor4()
        self.output_var_fixa90 = T.tensor4()
        self.output_var_fixa120 = T.tensor4()
        self.output_var_fixa180 = T.tensor4()
        self.output_var_fixa_C = T.tensor4()   
        
        self.output_var_wei = T.tensor4()
        
        self.inv_sigmaBCE = T.scalar()
        self.inv_sigmaKL = T.scalar()
        self.inv_sigmaCC = T.scalar()
        self.inv_sigmaNSS = T.scalar()
        self.inv_sigmaKL90 = T.scalar()
        self.inv_sigmaCC90 = T.scalar()
        self.inv_sigmaNSS90 = T.scalar()
        self.inv_sigmaKL120 = T.scalar()
        self.inv_sigmaCC120 = T.scalar()
        self.inv_sigmaNSS120 = T.scalar()
        self.inv_sigmaKL180 = T.scalar()
        self.inv_sigmaCC180 = T.scalar()
        self.inv_sigmaNSS180 = T.scalar()
        self.inv_sigmaKL_C = T.scalar()
        self.inv_sigmaCC_C = T.scalar()
        self.inv_sigmaNSS_C = T.scalar()
        
        self.meanBCE = T.scalar()
        self.meanKL = T.scalar()
        self.meanCC = T.scalar()
        self.meanNSS = T.scalar()
        
        
        self.adaptive_Glr = None
        self.adaptive_Dlr = None
        self.adaptive_Glr_in = T.scalar()
        self.adaptive_Dlr_in = T.scalar()