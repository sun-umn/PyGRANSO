from pygransoStruct import Options, sub_validators_struct
from dbg_print import dbg_print

def optionValidator( program_name, default_opts,sub_validators=None, sub_optname=None):
    # optionValidator:
    #    Helper object for performing common sanity and validity checks on 
    #    parameters provided by the user for any given program.
    
    # PRIVATE NESTED HELPER FUNCTIONS

    # Todo
    def checkAndSetSubValidators():
        # if sub_validators != None:
        #     cellfun(@checkAndSetSubValidator,fieldnames(sub_validators));    
        dbg_print('optionValidator checkAndSetSubValidators: TODO')


    # function checkAndSetSubValidator(name)
    #     assert( isfield(default_opts,name), id_str,                     ...
    #             'Sub-validator %s is missing from default_opts!',name   );
    #     assert( isa(sub_validators.(name),'function_handle'), id_str,   ...
    #             'Sub-validator %s must be a function handle!', name     );
    #     try 
    #         default_opts.(name) = sub_validators.(name)();
    #     catch err
    #         s = 'Sub-validator %s failed when requesting default values!';
    #         ME = MException(id_str,s,name);
    #         ME = addCause(ME,err);
    #         ME.throw();
    #     end
    # end


    ##########################################################################################
    ############################  Beginning of Main function #################################
    ##########################################################################################

    ov_str          = 'optionValidator'
    id_str          = ov_str + ':invalidUsage'
    unknown_str     = ov_str + ': it is not a recognized option.'

    #  CHECK INPUT OPTIONS
    # need error handler
    # assert( nargin > 1, id_str, notEnoughInputsMsg());

    #  need error handler here
    #  Mandatory Options
    assert isinstance(program_name,str),  id_str + "Input ''program_name'' must be a string!" 
    assert isinstance(default_opts, Options), id_str + "Input argument ''default_opts'' must be an Options class!"

    user_opts       = None
    opts            = default_opts

    # Optional subvalidator 3rd argument
    if sub_validators != None:
        assert isinstance(sub_validators, sub_validators_struct), id_str + "Input argument ''sub_validators'' must be an Options class!"
        checkAndSetSubValidators()
    
    err_id = program_name + ":invalidUserOption"

    # Optional suboptname 4th argument
    if sub_optname != None:
        dbg_print("optionValidator sub_optname: TODO")
    #     assert( ischar(sub_optname), id_str, 'Input ''sub_optname'' must be a string!'   );
    #     invalid_str = [ err_id ': .' sub_optname '.%s must be %s.' ];
    #     custom_str  = [ err_id ': .' sub_optname ': %s' ];
    # else
    #     invalid_str = [ err_id ': .%s must be %s.' ];
    #     custom_str  = [ err_id ': %s' ];
    
    dbg_print("optionValidator: currently assume all provided options are legal\n")
    # assertFn = lambda tf,varargin : assert(tf,err_id,varargin{:})
    

    return None