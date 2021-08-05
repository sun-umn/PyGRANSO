function validator = optionValidator(   program_name,   default_opts,   ...
                                        sub_validators, sub_optname     )
%   optionValidator:
%       Helper object for performing common sanity and validity checks on 
%       parameters provided by the user for any given program.
%
%   USAGE (an example):
%       % Set up the validator with all the default parameter values:
%       proc = optionValidator('your_program_name',struct_of_default_opts)
%       
%       % Initialize the validator with whatever parameters the user gave:
%       proc.setUserOpts(struct_of_user_options);
% 
%       % Validate and set opts.code_mode to an integer in {0,1,...,10}
%       % (This throws an error if the user's option does not meet this
%       % criteria.)  
%       proc.setIntegerInRange('code_mode',0,10)
%       
%       % Get all the options (default values plus validated user options)
%       validated_opts = proc.getValidatedOpts();
%
%   THROWS: 
%       optionValidator can throw two different types of errors:
%
%       1)  optionValidator:invalidUsage
%           This error is thrown when either the two necessary input
%           arguments are not provided OR if the user tries to validate/set
%           an option that doesn't exist in the default_opts struct.  Both
%           of these are developer-facing errors since they indicate that
%           the developer has improperly used optionValidator and needs to
%           correct their code.
%   
%       2)  YourProgramName:invalidUserOption
%           This error is thrown if a user-specified option is either empty
%           or fails to meet the validation criteria.  This is considered a
%           user-facing error. 
%
%       The reason for two types of errors is so that the developer can
%       have their software respond appropriately to both cases by
%       potentially catching these errors and having specific code handle
%       each case individually.
%
%   INPUT:
%       program_name       
%           A string to be used as a prefix create the error message ID:
%           YourProgramName:invalidOption
%   
%       default_opts
%           A struct containing all the parameter names as fields, with
%           each field set to the default value to be used for that
%           parameter.
%
%       sub_validators
%           A struct of option fieldnames that appear in default_opts which
%           are actually structs of options that need to be validated by
%           their own optionValidator-derived routines.  Each field must be
%           set to the function handle for its optionValidator-derived
%           validation routine.  sub_validators may be set to [] if there 
%           are no options requiring sub-validation.  For more information,
%           see optionValidator's setStructWithValidation routine.
% 
%       sub_optname     [optional]
%           A string to increase the specificity of all error messages.
%           Without this option, error messages for invalid parameter
%           values have the following format:
%          
%           YourProgramName:invalidUserOption: .option_name must blah
%  
%           By specifying sub_optname, the error messages become:
%
%           YourProgramName:invalidUserOption: .sub_optname.option_name ...
%           
%   OUTPUT:
%       An optionValidator object containing methods:
%       .setUserOpts(user_opts)   
%           Pass in the user's options (as a struct) to initialize the
%           processor.  Note that this does NOT set any value
%           Calling this resets all previously processed options.
%
%       .getDefaultOpts()
%           Returns a struct of the default parameters.  If any of the
%           parameters are actually substructs of options with validation
%           (see input argument sub_validators), these subfields will also
%           be populated with their default values returns by their
%           optionValidator-derived sub-validators.
%       
%       .getValidatedOpts()
%           Return the set of all processed/validated options, along with
%           the default values of the parameters that were not set/checked.
%           This MUST be called to get the validated options! 
%
%       .isSpecified(opt_name)
%           Returns true if the user has included the field opt_name in
%           their set of requested values.
%       
%       .validateValue(opt_name,validate_fn,error_msg)
%           Assert that the value stored in user_opts.(opt_name) causes
%           the validate_fn to return true; otherwise, the error message is
%           displayed.   Note that this does NOT set a value!
%       
%       .getValue(opt_name)
%           Get the latest value of the "opt_name" parameter; returns the
%           default value if no user option has been set yet.
%       
%       .validateAndSet(opt_name,validate_fn,error_msg)
%           Same as .validateValue but if the value is validated, then it
%           also sets the user's value.  This is useful if one has a
%           parameter that needs to be validated and set that has
%           conditions not provided by one of the builtin routines and/or
%           needs a custom error message.
%
%       The optionValidator object also has following builtin set routines
%       for validating common parameter types and then setting them.  All
%       functions take an opt_name string as their first argument, which
%       indicates which value, that is, user_opts.(opt_name), to be
%       validated and set.  If the value does not meet the criteria, an
%       error is thrown.
% 
%       NOTES:
%           1)  An invalid name of an option will cause an error to be 
%               thrown.  The names specified in user_opts must match those
%               in default_opts.
%
%           2)  User options set to the empty array [] are ignored
%               (assuming that they correspond to a valid option name)
%          
%           3)  Multiple conditions can be checked by calling all of the
%               appropriate set functions in a succession.  For example:
%                   
%               proc.setPositiveDefinite('hessian')
%               
%               only checks whether user_opts.hessian is numerically
%               positive definite but 
%       
%               proc.setRealAndFiniteValued('hessian')
%               proc.setPositiveDefinite('hessian')
%           
%               ensures that user_opts.hessian will not only be positive
%               definite but also only contain finite purely real values.
%
%       BASIC TYPE VALIDATIONS: 
%
%       Validate a basic type: logical, a struct, or a function handle
%       .setLogical(opt_name)
%       .setStruct(opt_name)
%       .setStructWithFields(opt_name,field1,field2,...)
%           Required input: 
%               opt_name and at least one field name (strings)
%           Output: 
%               a new optionValidator object for validating the fields of
%               the substruct.  user_opts.opt_name must contain all these
%               fields.
%       .setStructWithValidation(opt_name)
%           Checks the struct of options specified in field opt_name, using 
%           the optionValidator-derived sub-validator specified in
%           sub_validators.opt_name.
%       .setFunctionHandle(opt_name)
% 
%       INTEGER VALIDATIONS:
%
%       Validate an integer value (must be finite):
%       .setInteger(opt_name)
%       .setIntegerPositive(opt_name)
%       .setIntegerNegative(opt_name)
%       .setIntegerNonnegative(opt_name)
%       .setIntegerNonpositive(opt_name)
%
%       Validate number is in the integer range {min_value,...,max_value}
%       (min_value and max_value are allowed to be +/- infinity):
%       .setIntegerInRange(opt_name,min_value,max_value)
%       
% 
%       EXTENDED REAL VALIDATIONS: (+/- inf is okay, nan is not)
%   
%       Validate a real value:
%       .setReal(opt_name)
%       .setRealPositive(opt_name)
%       .setRealNegative(opt_name)
%       .setRealNonnegative(opt_name)
%       .setRealNonpositive(opt_name)
%       
%       Validate a real value in is an interval:
%       .setRealInIntervalOO(opt_name,a,b)      OO: open, open (a,b)
%       .setRealInIntervalOC(opt_name,a,b)      OC: open, closed (a,b]
%       .setRealInIntervalCO(opt_name,a,b)      CO: closed, open [a,b)
%       .setRealInIntervalCC(opt_name,a,b)      CC: closed, closed [a,b]
%
%       MATRIX/NUMERIC TYPE VALIDATIONS:
%
%       Validate all individual entries:
%       .setFiniteValued(opt_name)
%           All values must be finite (no nans/infs)
%       .setRealValued(opt_name)
%           All entries must be real (no or zero imaginary parts)
%       .setRealFiniteValued(opt_name)
%           All entries must be finite and real
%       .setNonZero(opt_name)
%           All values must not be zero (nans and infs are acceptable)
%
%       Validate dimensions of numeric type:
%       .setRow(opt_name)
%       .setRowDimensioned(opt_name,dim)
%           Must be a row vector of length dim
%       .setColumn(opt_name)
%       .setColumnDimensioned(opt_name,dim)
%           Must be a column vector of length dim
%       .setDimensioned(opt_name,M,N)
%           Matrix must have size M by N
%
%       Validate properties:     
%       .setSparse(opt_name)
%       .setPositiveDefinite(opt_name)
%           Matrix be positive definite, tested via chol()
%
%
%   For comments/bug reports, please visit the ROSTAPACK GitLab webpage:
%   https://gitlab.com/timmitchell/ROSTAPACK
%
%   optionValidator.m introduced in ROSTAPACK Version 1.0.
%
% =========================================================================
% |  optionValidator.m                                                    |
% |  Copyright (C) 2016-2018 Tim Mitchell                                 |
% |                                                                       |
% |  This file is originally from URTM.                                   |
% |                                                                       |
% |  URTM is free software: you can redistribute it and/or modify         |
% |  it under the terms of the GNU Affero General Public License as       |
% |  published by the Free Software Foundation, either version 3 of       |
% |  the License, or (at your option) any later version.                  |
% |                                                                       |
% |  URTM is distributed in the hope that it will be useful,              |
% |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
% |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
% |  GNU Affero General Public License for more details.                  |
% |                                                                       |
% |  You should have received a copy of the GNU Affero General Public     |
% |  License along with this program.  If not, see                        |
% |  <http://www.gnu.org/licenses/agpl.html>.                             |
% =========================================================================
%
% =========================================================================
% |  GRANSO: GRadient-based Algorithm for Non-Smooth Optimization         |
% |  Copyright (C) 2016 Tim Mitchell                                      |
% |                                                                       |
% |  This file is part of GRANSO.                                         |
% |                                                                       |
% |  GRANSO is free software: you can redistribute it and/or modify       |
% |  it under the terms of the GNU Affero General Public License as       |
% |  published by the Free Software Foundation, either version 3 of       |
% |  the License, or (at your option) any later version.                  |
% |                                                                       |
% |  GRANSO is distributed in the hope that it will be useful,            |
% |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
% |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
% |  GNU Affero General Public License for more details.                  |
% |                                                                       |
% |  You should have received a copy of the GNU Affero General Public     |
% |  License along with this program.  If not, see                        |
% |  <http://www.gnu.org/licenses/agpl.html>.                             |
% =========================================================================
    
    ov_str          = 'optionValidator';
    id_str          = [ov_str ':invalidUsage'];
    unknown_str     = [ov_str ': %s is not a recognized option.'];
    
    % CHECK INPUT OPTIONS
    assert( nargin > 1, id_str, notEnoughInputsMsg());
    
    % Mandatory Options
    assert( ischar(program_name), id_str,                       ...
            'Input ''program_name'' must be a string!'          );
    assert( isstruct(default_opts), id_str,                     ...
            'Input argument ''default_opts'' must be a struct!' );
        
    user_opts       = [];
    opts            = default_opts;
    
    % Optional subvalidator 3rd argument
    if nargin > 2 && ~isempty(sub_validators)
        assert( isstruct(sub_validators), id_str,                       ...
                'Input argument ''sub_validators'' must be a struct!'   );
        checkAndSetSubValidators()
    else
        sub_validators = [];
    end
    
    err_id = sprintf('%s:invalidUserOption',program_name);
    
    if nargin > 3
        % Optional suboptname 4th argument
        assert( ischar(sub_optname), id_str,                ...
                'Input ''sub_optname'' must be a string!'   );
        invalid_str = [ err_id ': .' sub_optname '.%s must be %s.' ];
        custom_str  = [ err_id ': .' sub_optname ': %s' ];
    else
        invalid_str = [ err_id ': .%s must be %s.' ];
        custom_str  = [ err_id ': %s' ];
    end
    assertFn        = @(tf,varargin) assert(tf,err_id,varargin{:});
    
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
   
    
    % PRIVATE NESTED HELPER FUNCTIONS

    % initialize optionProcessor with options from the user 
    function setUserOpts(some_user_opts)
        opts        = default_opts;
        assert( isstruct(some_user_opts),id_str,                        ...
                '%s.setUserOpts(s) requires that s is a struct.',ov_str );
        user_opts   = some_user_opts;
    end

    % get the default options
    function opts_out = getDefaultOpts()
        opts_out    = default_opts;
    end

    % get the processed/validated options
    function opts_out = getValidatedOpts()
        opts_out    = opts;
    end

    % get the latest version of a given value
    function value = getValue(opt_name)
        if isfield(opts,opt_name)
            value   = opts.(opt_name);
        else
            value   = [];
        end
    end
    
    % set the user's value if it meets the necessary conditions
    function validateAndSet(opt_name,validate_fn,error_msg)
        value = validateValue(opt_name,validate_fn,error_msg);
        if ~isempty(value)
            opts.(opt_name) = value;
        end
    end

    % checks the user's option for opt_name 
    % validate function is a function handle to check it is valid
    % error_msg is appended as the reason if the user's value is invalid
    function value = validateValue(opt_name,validate_fn,error_msg)
        value = [];
        % first make sure the name of the user option exists 
        assert(isfield(default_opts,opt_name),id_str,unknown_str,opt_name);
        if isfield(user_opts,opt_name)
            value = user_opts.(opt_name);
            % make sure the user's value is not empty 
            assertFn(~isempty(value),invalid_str,opt_name,'nonempty');
            % finally check the specific validation criteria
            assertFn(validate_fn(value),invalid_str,opt_name,error_msg);
        end
    end

    % checks whether the user specified a nonempty value for this parameter
    function tf = isSpecified(name)
        tf = isfield(user_opts,name) && ~isempty(user_opts.(name));
    end

    function customAssert(tf,error_msg)
        assertFn(tf,custom_str,error_msg);
    end

    % shortcut functions for common checks on parameters
    
    function setLogical(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) islogical(x) && isscalar(x),                           ...
            'a logical'                                                 );
    end

    function setFunctionHandle(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isa(x,'function_handle'),                              ...
            'a function handle'                                         );
    end

    function setStruct(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isstruct(x) && isscalar(x),                            ...
            'a struct'                                                  );
    end

    function sub_validator = setStructWithFields(name,varargin)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isstruct(x) && isscalar(x) && all(isfield(x,varargin)),...
            sprintf('a struct with fields: %s',strjoin(varargin,', '))  );
        
        sub_struct      = makeStructWithFields(varargin{:});
        user_sub_struct = user_opts.(name);
        sub_validator   = optionValidator(program_name,sub_struct,[],name);
        sub_validator.setUserOpts(user_sub_struct);
    end

    function setStructWithValidation(name)
        if isSpecified(name)
            setStruct(name);
            validated_sub_opts = sub_validators.(name)(user_opts.(name));
            opts.(name) = validated_sub_opts;
        end
    end
    
    function setInteger(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isAnInteger(x),                                        ...
            'an integer'                                                );
    end

    function setIntegerPositive(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isAnInteger(x) && x > 0,                               ...
            'a positive integer'                                        );
    end

    function setIntegerNegative(name)   
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isAnInteger(x) && x < 0,                               ...
            'a negative integer'                                        );
    end     

    function setIntegerNonnegative(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isAnInteger(x) && x >= 0,                              ...
            'a nonnegative integer'                                     );
    end

    function setIntegerNonpositive(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isAnInteger(x) && x <= 0,                              ...
            'a nonpositive integer'                                     );
    end

    function setIntegerInRange(name,l,r)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isAnInteger(x) && l <= x && x <= r,                    ...
            sprintf('an integer in {%g,...,%g}',l,r)                    );
    end

    function setReal(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isARealNumber(x),                                      ...
            'a real number'                                             );
    end

    function setRealPositive(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isARealNumber(x) && x > 0,                             ...
            'a positive real number'                                    );
    end

    function setRealNegative(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isARealNumber(x) && x < 0,                             ...
            'a negative real number'                                    );
    end

    function setRealNonnegative(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isARealNumber(x) && x >= 0,                            ...
            'a nonnegative real number'                                 );
    end

    function setRealNonpositive(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isARealNumber(x) && x <= 0,                            ...
            'a nonpositive real number'                                 );
    end

    function setRealInIntervalOO(name,l,r)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isARealNumber(x) && l < x && x < r,                    ...
            sprintf('in (%g,%g)',l,r)                                   );
    end

    function setRealInIntervalOC(name,l,r)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isARealNumber(x) && l < x && x <= r,                   ...
            sprintf('in (%g,%g]',l,r)                                   );
    end

    function setRealInIntervalCO(name,l,r)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isARealNumber(x) && l <= x && x < r,                   ...
            sprintf('in [%g,%g)',l,r)                                   );
    end

    function setRealInIntervalCC(name,l,r)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isARealNumber(x) && l <= x && x <= r,                  ...
            sprintf('in [%g,%g]',l,r)                                   );
    end

    function setRealValued(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isRealValued(x),                                       ...
            'real valued'                                               );
    end

    function setRealFiniteValued(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isRealValued(x) && isFiniteValued(x),                  ...
            'real and finite valued (no nans/infs allowed)'             );
    end

    function setFiniteValued(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isFiniteValued(x),                                     ...
            'finite valued (no nans/infs allowed)'                      );
    end

    function setNonZero(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) ~isZero(x),                                            ...
            'not identically equal to zero or only contain zeros'       );
    end

    function setRow(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isRow(x),                                              ...
            'a row vector'                                              );
    end

    function setRowDimensioned(name,dim)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isRow(x) && numel(x) == dim,                           ...
            sprintf('a row vector of length %d',dim)                    );
    end
      
    function setColumn(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isColumn(x),                                           ...
            'a column vector'                                           );
    end

    function setColumnDimensioned(name,dim)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isColumn(x) && numel(x) == dim,                        ...
            sprintf('a column vector of length %d',dim)                 );
    end

    function setDimensioned(name,m,n)                                          
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isMbyN(x,m,n),                                         ...
            sprintf('%d by %d',m,n)                                     );
    end

    function setSparse(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) issparse(x),                                           ...
            'a sparse matrix'                                           );
    end
      
    function setPositiveDefinite(name)
        validateAndSet(                                                 ...
            name,                                                       ...
            @(x) isFiniteValued(x) && isPositiveDefinite(x),            ...
            'a positive definite matrix'                                );
    end

    function checkAndSetSubValidators()
        if ~isempty(sub_validators)
            cellfun(@checkAndSetSubValidator,fieldnames(sub_validators));    
        end
    end

    function checkAndSetSubValidator(name)
        assert( isfield(default_opts,name), id_str,                     ...
                'Sub-validator %s is missing from default_opts!',name   );
        assert( isa(sub_validators.(name),'function_handle'), id_str,   ...
                'Sub-validator %s must be a function handle!', name     );
        try 
            default_opts.(name) = sub_validators.(name)();
        catch err
            s = 'Sub-validator %s failed when requesting default values!';
            ME = MException(id_str,s,name);
            ME = addCause(ME,err);
            ME.throw();
        end
    end
end

function msg = notEnoughInputsMsg()
msg = [                                                                 ...
'Not input arguments!  Valid ways of calling optionValidator are:\n'    ...
'optionValidator(program_name,default_opts)\n'                          ...
'optionValidator(program_name,default_opts,sub_validators)\n'           ...
'optionValidator(program_name,default_opts,sub_validators,suboptname)\n'];
end
