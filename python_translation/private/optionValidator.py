from pygransoStruct import Options, genral_struct, sub_validators_struct
from dbg_print import dbg_print
import types
from private.makeStructWithFields import makeStructWithFields
from private.isAnInteger import isAnInteger
from private.isARealNumber import isARealNumber
from private.isRealValued import isRealValued
from private.isFiniteValued import isFiniteValued
from private.isRow import isRow
from private.isColumn import isColumn
from private.isMbyN import isMbyN
from private.isPositiveDefinite import isPositiveDefinite

class oV:
    def __init__(self):
        pass

    def optionValidator( self, program_name, default_opts,sub_validators=None, sub_optname=None):
        """
        optionValidator:
           Helper object for performing common sanity and validity checks on 
           parameters provided by the user for any given program.
        """
        
        ##########################################################################################
        ############################  Beginning of Main function #################################
        ##########################################################################################

        self.program_name = program_name
        self.default_opts = default_opts
        self.sub_validators = sub_validators

        self.ov_str          = "optionValidator"
        id_str          = self.ov_str + ":invalidUsage"
        unknown_str     = self.ov_str + ": it is not a recognized option."

        #  CHECK INPUT OPTIONS
        # need error handler
        # assert( nargin > 1, id_str, notEnoughInputsMsg());

        #  need error handler here
        #  Mandatory Options
        assert isinstance(self.program_name,str),  self.id_str + "Input ''program_name'' must be a string!" 
        assert isinstance(self.default_opts, Options), self.id_str + "Input argument ''default_opts'' must be an Options class!"

        self.user_opts       = None
        self.opts            = self.default_opts

        # Optional subvalidator 3rd argument
        if self.sub_validators != None:
            assert isinstance(self.sub_validators, sub_validators_struct), self.id_str + "Input argument ''sub_validators'' must be an Options class!"
            self.checkAndSetSubValidators()
        
        err_id = self.program_name + ":invalidUserOption"

        # Optional suboptname 4th argument
        if sub_optname != None:
            # dbg_print("optionValidator sub_optname: TODO")
            assert isinstance(sub_optname,str), self.id_str + "Input ''sub_optname'' must be a string!"
            self.invalid_str = err_id + ": ." + sub_optname + ".%s must be %s." 
            self.custom_str  = err_id + ": ." + sub_optname + ": %s"
        else:
            self.invalid_str = err_id + ": .%s must be %s."
            self.custom_str  = err_id + ": %s"
        
        # dbg_print("optionValidator: currently assume all provided options are legal\n")
        # assertFn = lambda tf,varargin : assert tf, err_id + varargin
        
        validator = genral_struct()
        setattr(validator, "setUserOpts", lambda some_user_opts : self.setUserOpts(some_user_opts))
        setattr(validator, "assert", lambda tf, error_msg : self.customAssert(tf, error_msg))
        
        validator = struct(                                             ...
        'setUserOpts',                  @setUserOpts,               ...
        'assert',                       @customAssert,              ...
        'isSpecified',                  @isSpecified,               ...
        'validateValue',                @validateValue,             ...
        'validateAndSet',               @validateAndSet,            ...
        'getValue',                     @getValue,                  ...
        'getDefaultOpts',               @getDefaultOpts,            ...
        'getValidatedOpts',             @getValidatedOpts,          ...
        'setLogical',                   @setLogical,                ...
        'setStruct',                    @setStruct,                 ...
        'setStructWithFields',          @setStructWithFields,       ... 
        'setStructWithValidation',      @setStructWithValidation,   ...
        'setFunctionHandle',            @setFunctionHandle,         ...
        'setInteger',                   @setInteger,                ...
        'setIntegerPositive',           @setIntegerPositive,        ...
        'setIntegerNegative',           @setIntegerNegative,        ...
        'setIntegerNonnegative',        @setIntegerNonnegative,     ...
        'setIntegerNonpositive',        @setIntegerNonpositive,     ...
        'setIntegerInRange',            @setIntegerInRange,         ...
        'setReal',                      @setReal,                   ...
        'setRealPositive',              @setRealPositive,           ...
        'setRealNegative',              @setRealNegative,           ...
        'setRealNonnegative',           @setRealNonnegative,        ...
        'setRealNonpositive',           @setRealNonpositive,        ...
        'setRealInIntervalOO',          @setRealInIntervalOO,       ...
        'setRealInIntervalOC',          @setRealInIntervalOC,       ...
        'setRealInIntervalCO',          @setRealInIntervalCO,       ...
        'setRealInIntervalCC',          @setRealInIntervalCC,       ...
        'setFiniteValued',              @setFiniteValued,           ...
        'setRealValued',                @setRealValued,             ...
        'setRealFiniteValued',          @setRealFiniteValued,       ...
        'setNonZero',                   @setNonZero,                ...
        'setRow',                       @setRow,                    ...
        'setRowDimensioned',            @setRowDimensioned,         ...
        'setColumn',                    @setColumn,                 ...
        'setColumnDimensioned',         @setColumnDimensioned,      ...
        'setDimensioned',               @setDimensioned,            ...
        'setSparse',                    @setSparse,                 ...
        'setPositiveDefinite',          @setPositiveDefinite        );

        return validator


    # PRIVATE NESTED HELPER FUNCTIONS

    #  initialize optionProcessor with options from the user 
    def setUserOpts(self,some_user_opts):
        self.opts        = self.default_opts.copy()
        assert isinstance(some_user_opts, Options), self.id_str + "%s.setUserOpts(s) requires that s is a struct." % self.ov_str 
        self.user_opts   = some_user_opts.copy()
        return

    #  get the default options
    def getDefaultOpts(self):
        opts_out    = self.default_opts.copy()
        return opts_out

    #  get the processed/validated options
    def getValidatedOpts(self):
        opts_out    = self.opts
        return opts_out

    #  get the latest version of a given value
    def getValue(self,opt_name):
        if hasattr(self.opts,opt_name):
            value   =  getattr(self.opts,opt_name)  
        else:
            value   = None
        
        return value
    
    #  set the user's value if it meets the necessary conditions
    def validateAndSet(self,opt_name,validate_fn,error_msg):
        value = self.validateValue(opt_name,validate_fn,error_msg)
        if value != None:
            setattr(self.opts,opt_name, value)  

    #  checks the user's option for opt_name 
    #  validate function is a function handle to check it is valid
    #  error_msg is appended as the reason if the user's value is invalid
    def validateValue(self,opt_name,validate_fn,error_msg):
        value = None
        #  first make sure the name of the user option exists 
        assert hasattr(self.default_opts,opt_name),self.id_str + self.unknown_str + opt_name
        if hasattr(self.user_opts,opt_name):
            value = getattr(self.user_opts, opt_name)  
            #  make sure the user's value is not empty 
            assert value != None, self.invalid_str + opt_name + " nonempty"
            #  finally check the specific validation criteria
            assert validate_fn(value) ,self.invalid_str + opt_name+error_msg
        
        return value

    #  checks whether the user specified a nonempty value for this parameter
    def isSpecified(self,name):
        tf = hasattr(self.user_opts,name) and getattr(self.user_opts,name) != None
        return tf

    def customAssert(self,tf,error_msg):
        assert tf, self.custom_str + error_msg
        return

    #  shortcut functions for common checks on parameters
    
    def setLogical(self,name):
        self.validateAndSet(name, lambda x: isinstance(x,int), "a logical" )
        return

    def setFunctionHandle(self,name):
        self.validateAndSet( name, lambda x: isinstance(x,types.LambdaType), 'a function handle' )
        return

    def setStruct(self,name):
        self.validateAndSet( name, lambda x: isinstance(x,genral_struct) , "a struct" )
        return

    def setStructWithFields(self,name,varargin):
        self.validateAndSet( name, lambda x: isinstance(x,genral_struct) and [hasattr(x,field) for field in varargin], "a struct with fields: %s" % ", ".join(varargin) )  
        
        sub_struct      = makeStructWithFields(varargin)
        user_sub_struct = getattr(self.user_opts, name).copy()
        sub_validator   = self.optionValidator(self.program_name,sub_struct,None,name)
        sub_validator.setUserOpts(user_sub_struct)
        return sub_validator

    def setStructWithValidation(self,name):
        if self.isSpecified(name):
            self.setStruct(name)
            validated_sub_opts = getattr(self.sub_validators,name)(getattr(self.user_opts,name))
            getattr(self.opts,name) = validated_sub_opts
        return 
    
    def setInteger(self,name):
        self.validateAndSet( name, lambda x: isAnInteger(x), "an integer" )
        

    def setIntegerPositive(self,name):
        self.validateAndSet( name, lambda x: isAnInteger(x) and x > 0, "a positive integer" )
        return

    def setIntegerNegative(self,name):   
        self.validateAndSet( name, lambda x: isAnInteger(x) and x < 0, "a negative integer" )
        return     

    def setIntegerNonnegative(self,name):
        self.validateAndSet( name, lambda x: isAnInteger(x) and x >= 0, "a nonnegative integer" )
        return

    def setIntegerNonpositive(self,name):
        self.validateAndSet( name, lambda x: isAnInteger(x) and x <= 0, "a nonpositive integer" )
        return

    def setIntegerInRange(self,name,l,r):
        self.validateAndSet( name, lambda x: isAnInteger(x) and l <= x and x <= r, "an integer in {%g,...,%g}",(l,r) )
        return 

    def setReal(self,name):
        self.validateAndSet( name, lambda x: isARealNumber(x), 'a real number')
        return

    def setRealPositive(self,name):
        self.validateAndSet( name, lambda x: isARealNumber(x) and x > 0, "a positive real number")
        return

    def setRealNegative(self,name):
        self.validateAndSet( name, lambda x: isARealNumber(x) and x < 0, "a negative real number")
        return

    def setRealNonnegative(self,name):
        self.validateAndSet( name, lambda x: isARealNumber(x) and x >= 0, "a nonnegative real number" )
        return

    def setRealNonpositive(self,name):
        self.validateAndSet( name, lambda x: isARealNumber(x) and x <= 0, "a nonpositive real number")
        return

    def setRealInIntervalOO(self,name,l,r):
        self.validateAndSet( name, lambda x: isARealNumber(x) and l < x and x < r, "in (%g,%g)"%(l,r))
        return

    def setRealInIntervalOC(self,name,l,r):
        self.validateAndSet( name, lambda x: isARealNumber(x) and l < x and x <= r, "in (%g,%g]"%(l,r))
        return

    def setRealInIntervalCO(self,name,l,r):
        self.validateAndSet( name, lambda x: isARealNumber(x) and l <= x and x < r, "in [%g,%g)"%(l,r))
        return

    def setRealInIntervalCC(self,name,l,r):
        self.validateAndSet( name, lambda x: isARealNumber(x) and l <= x and x <= r, "in [%g,%g]"%(l,r) )
        return

    def setRealValued(self,name):
        self.validateAndSet( name, lambda x: isRealValued(x), "real valued")
        return

    def setRealFiniteValued(self,name):
        self.validateAndSet( name, lambda x: isRealValued(x) and isFiniteValued(x),  "real and finite valued (no nans/infs allowed)")
        return

    def setFiniteValued(self,name):
        self.validateAndSet( name, lambda x: isFiniteValued(x), "finite valued (no nans/infs allowed)")
        return

    def setNonZero(self,name):
        self.validateAndSet( name, lambda x: not isZero(x), "not identically equal to zero or only contain zeros")
        return

    def setRow(self,name):
        self.validateAndSet( name, lambda x: isRow(x), "a row vector" )
        return

    def setRowDimensioned(self,name,dim):
        self.validateAndSet( name, lambda x: isRow(x) and x.size == dim, "a row vector of length %d"%dim )
        return
      
    def setColumn(self,name):
        self.validateAndSet( name,lambda x: isColumn(x), "a column vector")
        return

    def setColumnDimensioned(self,name,dim):
        self.validateAndSet( name, lambda x: isColumn(x) and x.size == dim, "a column vector of length %d"%dim)
        return

    def setDimensioned(self,name,m,n):                                          
        self.validateAndSet( name, lambda x: isMbyN(x,m,n), "%d by %d"%(m,n) )
        return

    def setSparse(self,name):
        dbg_print("TODO: optionValidator.setSparse")
        # self.validateAndSet(name, lambda x: issparse(x), "a sparse matrix")
        return
      
    def setPositiveDefinite(self,name):
        self.validateAndSet( name, lambda x: isFiniteValued(x) and isPositiveDefinite(x), "a positive definite matrix")
        return

    # Todo
    def checkAndSetSubValidators(self):
        if self.sub_validators != None:
            # cellfun(@checkAndSetSubValidator,fieldnames(sub_validators)); 
            for name in sub_validators_struct.__dict__:
                self.checkAndSetSubValidator(name)
               
        # dbg_print('optionValidator checkAndSetSubValidators: TODO')


    def checkAndSetSubValidator(self,name):
        assert hasattr(self.default_opts,name), self.id_str + "Sub-validator %s is missing from default_opts!"%name
        assert isinstance(  getattr(self.sub_validators,name), types.FunctionType), self.id_str + "Sub-validator %s must be a function handle!"%+ name
        try: 
            setattr(self.default_opts,name,getattr(self.sub_validators,name) )   
        except Exception as e:
            print(e)
            s = "Sub-validator %s failed when requesting default values!"
            print(s%name)
        return