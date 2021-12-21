from ncvxStruct import Options, GeneralStruct, sub_validators_struct
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
from private.isString import isString
from private.isRestartData import isRestartData
import numpy as np
import copy
import torch
import traceback,sys

class oV:
    def __init__(self):
        pass

    def optionValidator( self, program_name, default_opts,sub_validators=None, sub_optname=None):
        """
        optionValidator:
           Helper object for performing common sanity and validity checks on 
           parameters provided by the user for any given program.


            USAGE (an example):
                % Set up the validator with all the default parameter values:
                proc = optionValidator('your_program_name',struct_of_default_opts)
                
                % Initialize the validator with whatever parameters the user gave:
                proc.setUserOpts(struct_of_user_options);
        
                % Validate and set opts.code_mode to an integer in {0,1,...,10}
                % (This throws an error if the user's option does not meet this
                % criteria.)  
                proc.setIntegerInRange('code_mode',0,10)
                
                % Get all the options (default values plus validated user options)
                validated_opts = proc.getValidatedOpts();
        
            THROWS: 
                optionValidator can throw two different types of errors:
        
                1)  optionValidator:invalidUsage
                    This error is thrown when either the two necessary input
                    arguments are not provided OR if the user tries to validate/set
                    an option that doesn't exist in the default_opts struct.  Both
                    of these are developer-facing errors since they indicate that
                    the developer has improperly used optionValidator and needs to
                    correct their code.
            
                2)  YourProgramName:invalidUserOption
                    This error is thrown if a user-specified option is either empty
                    or fails to meet the validation criteria.  This is considered a
                    user-facing error. 
        
                The reason for two types of errors is so that the developer can
                have their software respond appropriately to both cases by
                potentially catching these errors and having specific code handle
                each case individually.
        
            INPUT:
                program_name       
                    A string to be used as a prefix create the error message ID:
                    YourProgramName:invalidOption
            
                default_opts
                    A struct containing all the parameter names as fields, with
                    each field set to the default value to be used for that
                    parameter.
        
                sub_validators
                    A struct of option fieldnames that appear in default_opts which
                    are actually structs of options that need to be validated by
                    their own optionValidator-derived routines.  Each field must be
                    set to the function handle for its optionValidator-derived
                    validation routine.  sub_validators may be set to [] if there 
                    are no options requiring sub-validation.  For more information,
                    see optionValidator's setStructWithValidation routine.
        
                sub_optname     [optional]
                    A string to increase the specificity of all error messages.
                    Without this option, error messages for invalid parameter
                    values have the following format:
                
                    YourProgramName:invalidUserOption: .option_name must blah
        
                    By specifying sub_optname, the error messages become:
        
                    YourProgramName:invalidUserOption: .sub_optname.option_name ...
                    
            OUTPUT:
                An optionValidator object containing methods:
                .setUserOpts(user_opts)   
                    Pass in the user's options (as a struct) to initialize the
                    processor.  Note that this does NOT set any value
                    Calling this resets all previously processed options.
        
                .getDefaultOpts()
                    Returns a struct of the default parameters.  If any of the
                    parameters are actually substructs of options with validation
                    (see input argument sub_validators), these subfields will also
                    be populated with their default values returns by their
                    optionValidator-derived sub-validators.
                
                .getValidatedOpts()
                    Return the set of all processed/validated options, along with
                    the default values of the parameters that were not set/checked.
                    This MUST be called to get the validated options! 
        
                .isSpecified(opt_name)
                    Returns true if the user has included the field opt_name in
                    their set of requested values.
                
                .validateValue(opt_name,validate_fn,error_msg)
                    Assert that the value stored in user_opts.(opt_name) causes
                    the validate_fn to return true; otherwise, the error message is
                    displayed.   Note that this does NOT set a value!
                
                .getValue(opt_name)
                    Get the latest value of the "opt_name" parameter; returns the
                    default value if no user option has been set yet.
                
                .validateAndSet(opt_name,validate_fn,error_msg)
                    Same as .validateValue but if the value is validated, then it
                    also sets the user's value.  This is useful if one has a
                    parameter that needs to be validated and set that has
                    conditions not provided by one of the builtin routines and/or
                    needs a custom error message.
        
                The optionValidator object also has following builtin set routines
                for validating common parameter types and then setting them.  All
                functions take an opt_name string as their first argument, which
                indicates which value, that is, user_opts.(opt_name), to be
                validated and set.  If the value does not meet the criteria, an
                error is thrown.
        
                NOTES:
                    1)  An invalid name of an option will cause an error to be 
                        thrown.  The names specified in user_opts must match those
                        in default_opts.
        
                    2)  User options set to the empty array [] are ignored
                        (assuming that they correspond to a valid option name)
                
                    3)  Multiple conditions can be checked by calling all of the
                        appropriate set functions in a succession.  For example:
                            
                        proc.setPositiveDefinite('hessian')
                        
                        only checks whether user_opts.hessian is numerically
                        positive definite but 
                
                        proc.setRealAndFiniteValued('hessian')
                        proc.setPositiveDefinite('hessian')
                    
                        ensures that user_opts.hessian will not only be positive
                        definite but also only contain finite purely real values.
        
                BASIC TYPE VALIDATIONS: 
        
                Validate a basic type: logical, a struct, or a function handle
                .setLogical(opt_name)
                .setStruct(opt_name)
                .setStructWithFields(opt_name,field1,field2,...)
                    Required input: 
                        opt_name and at least one field name (strings)
                    Output: 
                        a new optionValidator object for validating the fields of
                        the substruct.  user_opts.opt_name must contain all these
                        fields.
                .setStructWithValidation(opt_name)
                    Checks the struct of options specified in field opt_name, using 
                    the optionValidator-derived sub-validator specified in
                    sub_validators.opt_name.
                .setFunctionHandle(opt_name)
        
                INTEGER VALIDATIONS:
        
                Validate an integer value (must be finite):
                .setInteger(opt_name)
                .setIntegerPositive(opt_name)
                .setIntegerNegative(opt_name)
                .setIntegerNonnegative(opt_name)
                .setIntegerNonpositive(opt_name)
        
                Validate number is in the integer range {min_value,...,max_value}
                (min_value and max_value are allowed to be +/- infinity):
                .setIntegerInRange(opt_name,min_value,max_value)
                
        
                EXTENDED REAL VALIDATIONS: (+/- inf is okay, nan is not)
            
                Validate a real value:
                .setReal(opt_name)
                .setRealPositive(opt_name)
                .setRealNegative(opt_name)
                .setRealNonnegative(opt_name)
                .setRealNonpositive(opt_name)
                
                Validate a real value in is an interval:
                .setRealInIntervalOO(opt_name,a,b)      OO: open, open (a,b)
                .setRealInIntervalOC(opt_name,a,b)      OC: open, closed (a,b]
                .setRealInIntervalCO(opt_name,a,b)      CO: closed, open [a,b)
                .setRealInIntervalCC(opt_name,a,b)      CC: closed, closed [a,b]
        
                MATRIX/NUMERIC TYPE VALIDATIONS:
        
                Validate all individual entries:
                .setFiniteValued(opt_name)
                    All values must be finite (no nans/infs)
                .setRealValued(opt_name)
                    All entries must be real (no or zero imaginary parts)
                .setRealFiniteValued(opt_name)
                    All entries must be finite and real
                .setNonZero(opt_name)
                    All values must not be zero (nans and infs are acceptable)
        
                Validate dimensions of numeric type:
                .setRow(opt_name)
                .setRowDimensioned(opt_name,dim)
                    Must be a row vector of length dim
                .setColumn(opt_name)
                .setColumnDimensioned(opt_name,dim)
                    Must be a column vector of length dim
                .setDimensioned(opt_name,M,N)
                    Matrix must have size M by N
        
                Validate properties:     
                .setSparse(opt_name)
                .setPositiveDefinite(opt_name)
                    Matrix be positive definite, tested via chol() 

            If you publish work that uses or refers to NCVX, please cite both
            NCVX and GRANSO paper:

            [1] Buyun Liang, and Ju Sun. 
                NCVX: A User-Friendly and Scalable Package for Nonconvex 
                Optimization in Machine Learning. arXiv preprint arXiv:2111.13984 (2021).
                Available at https://arxiv.org/abs/2111.13984

            [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton 
                A BFGS-SQP method for nonsmooth, nonconvex, constrained 
                optimization and its evaluation using relative minimization 
                profiles, Optimization Methods and Software, 32(1):148-181, 2017.
                Available at https://dx.doi.org/10.1080/10556788.2016.1208749
                
            Change Log:
                optionValidator.m introduced in GRANSO Version 1.0
                
                Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                    optionValidator.py is translated from optionValidator.m in GRANSO Version 1.6.4. 

            For comments/bug reports, please visit the NCVX webpage:
            https://github.com/sun-umn/NCVX
            
            NCVX Version 1.0.0, 2021, see AGPL license info below.

            =========================================================================
            |  optionValidator.m                                                    |
            |  Copyright (C) 2016 Tim Mitchell                                      |
            |                                                                       |
            |  This file is originally from URTM.                                   |
            |                                                                       |
            |  URTM is free software: you can redistribute it and/or modify         |
            |  it under the terms of the GNU Affero General Public License as       |
            |  published by the Free Software Foundation, either version 3 of       |
            |  the License, or (at your option) any later version.                  |
            |                                                                       |
            |  URTM is distributed in the hope that it will be useful,              |
            |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
            |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
            |  GNU Affero General Public License for more details.                  |
            |                                                                       |
            |  You should have received a copy of the GNU Affero General Public     |
            |  License along with this program.  If not, see                        |
            |  <http://www.gnu.org/licenses/agpl.html>.                             |
            =========================================================================

            =========================================================================
            |  GRANSO: GRadient-based Algorithm for Non-Smooth Optimization         |
            |  Copyright (C) 2016 Tim Mitchell                                      |
            |                                                                       |
            |  This file is translated from GRANSO.                                 |
            |                                                                       |
            |  GRANSO is free software: you can redistribute it and/or modify       |
            |  it under the terms of the GNU Affero General Public License as       |
            |  published by the Free Software Foundation, either version 3 of       |
            |  the License, or (at your option) any later version.                  |
            |                                                                       |
            |  GRANSO is distributed in the hope that it will be useful,            |
            |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
            |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
            |  GNU Affero General Public License for more details.                  |
            |                                                                       |
            |  You should have received a copy of the GNU Affero General Public     |
            |  License along with this program.  If not, see                        |
            |  <http://www.gnu.org/licenses/agpl.html>.                             |
            =========================================================================

            =========================================================================
            |  NCVX (NonConVeX): A User-Friendly and Scalable Package for           |
            |  Nonconvex Optimization in Machine Learning.                          |
            |                                                                       |
            |  Copyright (C) 2021 Buyun Liang                                       |
            |                                                                       |
            |  This file is part of NCVX.                                           |
            |                                                                       |
            |  NCVX is free software: you can redistribute it and/or modify         |
            |  it under the terms of the GNU Affero General Public License as       |
            |  published by the Free Software Foundation, either version 3 of       |
            |  the License, or (at your option) any later version.                  |
            |                                                                       |
            |  GRANSO is distributed in the hope that it will be useful,            |
            |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
            |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
            |  GNU Affero General Public License for more details.                  |
            |                                                                       |
            |  You should have received a copy of the GNU Affero General Public     |
            |  License along with this program.  If not, see                        |
            |  <http://www.gnu.org/licenses/agpl.html>.                             |
            =========================================================================
        """
        
        ##########################################################################################
        ############################  Beginning of Main function #################################
        ##########################################################################################

        self.program_name = program_name
        self.default_opts = default_opts
        self.sub_validators = sub_validators

        self.ov_str          = "optionValidator"
        self.id_str          = self.ov_str + ":invalidUsage"
        self.unknown_str     = self.ov_str + ": it is not a recognized option."

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
            assert isinstance(sub_optname,str), self.id_str + "Input ''sub_optname'' must be a string!"
            self.invalid_str = err_id + ": ." + sub_optname + ".%s must be %s." 
            self.custom_str  = err_id + ": ." + sub_optname + ": %s"
        else:
            self.invalid_str = err_id + ": .%s must be %s."
            self.custom_str  = err_id + ": %s"
        
        validator = GeneralStruct()
        setattr(validator, "setUserOpts", lambda some_user_opts : self.setUserOpts(some_user_opts))
        setattr(validator, "assert", lambda tf, error_msg : self.customAssert(tf, error_msg))
        setattr(validator, "isSpecified", lambda name : self.isSpecified(name))
        setattr(validator, "validateValue", lambda opt_name, validate_fn, error_msg : self.validateValue(opt_name, validate_fn, error_msg))
        setattr(validator, "validateAndSet", lambda opt_name, validate_fn, error_msg : self.validateAndSet(opt_name, validate_fn, error_msg))
        setattr(validator, "getValue", lambda opt_name : self.getValue(opt_name))
        setattr(validator, "getDefaultOpts", lambda  : self.getDefaultOpts())
        setattr(validator, "getValidatedOpts", lambda : self.getValidatedOpts( ))
        setattr(validator, "setLogical", lambda name : self.setLogical(name))
        setattr(validator, "setStruct", lambda name : self.setStruct(name))
        setattr(validator, "setStructWithFields", lambda name, varargin : self.setStructWithFields(name, varargin))
        setattr(validator, "setStructWithValidation", lambda name : self.setStructWithValidation(name))
        setattr(validator, "setFunctionHandle", lambda name : self.setFunctionHandle(name))
        setattr(validator, "setInteger", lambda name : self.setInteger(name))
        setattr(validator, "setIntegerPositive", lambda  name: self.setIntegerPositive(name))
        setattr(validator, "setIntegerNegative", lambda name: self.setIntegerNegative(name ))
        setattr(validator, "setIntegerNonnegative", lambda name : self.setIntegerNonnegative(name))
        setattr(validator, "setIntegerNonpositive", lambda name : self.setIntegerNonpositive(name))
        setattr(validator, "setIntegerInRange", lambda name, l, r : self.setIntegerInRange(name, l, r))
        setattr(validator, "setReal", lambda  name: self.setReal(name))
        setattr(validator, "setRealPositive", lambda name: self.setRealPositive(name ))
        setattr(validator, "setRealNegative", lambda name : self.setRealNegative(name))
        setattr(validator, "setRealNonnegative", lambda name : self.setRealNonnegative(name))
        setattr(validator, "setRealNonpositive", lambda name : self.setRealNonpositive(name))
        setattr(validator, "setRealInIntervalOO", lambda  name, l, r: self.setRealInIntervalOO(name, l, r))
        setattr(validator, "setRealInIntervalOC", lambda name, l, r: self.setRealInIntervalOC(name, l, r ))
        setattr(validator, "setRealInIntervalCO", lambda  name, l, r: self.setRealInIntervalCO(name, l, r))
        setattr(validator, "setRealInIntervalCC", lambda name, l, r: self.setRealInIntervalCC(name, l, r ))
        setattr(validator, "setFiniteValued", lambda  name: self.setFiniteValued(name))
        setattr(validator, "setRealValued", lambda name: self.setRealValued(name ))
        setattr(validator, "setRealFiniteValued", lambda name : self.setRealFiniteValued(name))
        setattr(validator, "setNonZero", lambda name : self.setNonZero(name))
        setattr(validator, "setRow", lambda name : self.setRow(name))
        setattr(validator, "setRowDimensioned", lambda  name, dim: self.setRowDimensioned(name, dim))
        setattr(validator, "setColumn", lambda name: self.setColumn(name ))
        setattr(validator, "setColumnDimensioned", lambda name, dim : self.setColumnDimensioned(name, dim))
        setattr(validator, "setDimensioned", lambda name, m, n : self.setDimensioned(name, m, n))
        setattr(validator, "setSparse", lambda name : self.setSparse(name))
        setattr(validator, "setPositiveDefinite", lambda name : self.setPositiveDefinite(name))

        setattr(validator, "setString", lambda name : self.setString(name))
        setattr(validator,"setRestartData",lambda name: self.setRestartData(name))

        return validator


    # PRIVATE NESTED HELPER FUNCTIONS

    #  initialize optionProcessor with options from the user 
    def setUserOpts(self,some_user_opts):
        self.opts        = copy.deepcopy(self.default_opts) 
        assert isinstance(some_user_opts, Options), self.id_str + "%s.setUserOpts(s) requires that s is a struct." % self.ov_str 
        self.user_opts   = copy.deepcopy(some_user_opts)
        return

    #  get the default options
    def getDefaultOpts(self):
        opts_out    = copy.deepcopy(self.default_opts) 
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
        if np.any(value != None):
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
            assert np.any(value != None), self.invalid_str + opt_name + " nonempty"
            #  finally check the specific validation criteria
            assert validate_fn(value) ,self.invalid_str %(opt_name,error_msg)
        
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
        self.validateAndSet( name, lambda x: isinstance(x,GeneralStruct) , "a struct" )
        return

    def setStructWithFields(self,name,varargin):
        self.validateAndSet( name, lambda x: isinstance(x,GeneralStruct) or isinstance(x,Options) and [hasattr(x,field) for field in varargin], "a struct with fields: %s" % ", ".join(varargin) )  
        
        sub_struct      = makeStructWithFields(varargin)
        # user_sub_struct = copy.deepcopy(getattr(self.user_opts, name))
        user_sub_struct = getattr(self.user_opts, name)
        sub_validator   = self.optionValidator(self.program_name,sub_struct,None,name)
        sub_validator.setUserOpts(user_sub_struct)
        return sub_validator

    def setStructWithValidation(self,name):
        if self.isSpecified(name):
            self.setStruct(name)
            validated_sub_opts = getattr(self.sub_validators,name)(getattr(self.user_opts,name))
            setattr(self.opts,name,validated_sub_opts)
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
        self.validateAndSet( name, lambda x: isAnInteger(x) and l <= x and x <= r, "an integer in {%g,...,%g}"%(l,r) )
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
        self.validateAndSet( name, lambda x: isColumn(x) and torch.numel(x) == dim, "a column vector of length %d"%dim)
        return

    def setDimensioned(self,name,m,n):                                          
        self.validateAndSet( name, lambda x: isMbyN(x,m,n), "%d by %d"%(m,n) )
        return

    def setRestartData(self,name):                                          
        self.validateAndSet( name, lambda x: isRestartData(x), "lbfgs_warm_start required form {'S':matrix, 'Y':matrix, 'rho':row vector, 'gamma':scalar}" )
        return

    def setSparse(self,name):
        # optionValidator.setSparse NOT used
        # self.validateAndSet(name, lambda x: issparse(x), "a sparse matrix")
        return
      
    def setPositiveDefinite(self,name):
        self.validateAndSet( name, lambda x: isFiniteValued(x) and isPositiveDefinite(x), "a positive definite matrix")
        return

    def setString(self,name):
        self.validateAndSet( name,lambda x: isString(x), "a string")
        return

    def checkAndSetSubValidators(self):
        if self.sub_validators != None:
            for name in sub_validators_struct.__dict__:
                self.checkAndSetSubValidator(name)


    def checkAndSetSubValidator(self,name):
        assert hasattr(self.default_opts,name), self.id_str + "Sub-validator %s is missing from default_opts!"%name
        assert isinstance(  getattr(self.sub_validators,name), types.FunctionType), self.id_str + "Sub-validator %s must be a function handle!"%+ name
        try: 
            setattr(self.default_opts,name,getattr(self.sub_validators,name) )   
        except Exception as e:
            print(traceback.format_exc())
            s = "Sub-validator %s failed when requesting default values!"
            print(s%name)
            sys.exit()
        return