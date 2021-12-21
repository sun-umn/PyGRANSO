from ncvxStruct import GeneralStruct
from private import copyrightNotice, ncvxConstants as pC, ncvxPrinterColumns as pPC, printMessageBox as pMB
from private.tablePrinter import tP
from private.nDigitsInWholePart import nDigitsInWholePart
from private.copyrightNotice import copyrightNotice

class pgP:
    def __init__(self):
        pass

    def ncvxPrinter(self,opts,n,n_ineq,n_eq):
        """    
        ncvxPrinter:
            Object for handling printing out info for each iteration and
            messages.

            INPUT:
                opts                        
                    Struct of necessary parameters from GRANSO that the printer
                    needs to know about, for keeping the iteration printing nicely
                    aligned and what print options are to be used. 
                .print_ascii
                .print_use_orange
                .print_width  
                .maxit       
                .limited_mem_size
                .linesearch_reattempts
                .max_random_attempts
                .max_fallback_level
                .ngrad

                For more information on the above parameters, see gransoOptions and
                gransoOptionsAdvanced.

                n 
                    number of optimization variables.

                ineq_constraints    
                    nonnegative number indicating the number of inequality
                    constraints that are present.

                eq_constraints    
                    nonnegative integer indicating the number of equality
                    constraints that are present.

            OUTPUT:
                An "object", a struct containing the following functions for
                printing tasks:
                .msg                
                    Prints a message inside the table; the full width is available
                    and cell arrays can be used to print multiple lines.
                .msgWidth          
                    Returns the number of chars wide the message area.
                .init               
                    For printing the initial info at x0.  Requires input arguments: 
                        penfn_at_x      state of the penalty function.  For more
                                        info, see makePenaltyFunction.m
                        stat_value      stationarity measure value
                        n_qps           number of QPs solved
                .iter           
                    Print the info for the current iterate.  Requires arguments:
                        iter
                        penfn_at_x      state of the penalty function. For more
                                        info, see makePenaltyFunction.m
                        sd_code         integer code indicating which search
                                        direction was used.
                        random_attempts number of random attempts to produce search
                                        direction (this is generally zero)
                        ls_evals        number of functions evaluations incurred in
                                        the line search to find an acceptable
                                        next iterate
                        alpha           step length taken to get this next iterate
                        n_grads         number of gradients used in computing the 
                                        approximate stationarity measure
                        stat_value      value of the (approx) stationarity measure
                        n_qps           number of QP solves incurred to compute 
                                        the (approx) stationarity measure    
                .summary
                    Prints the objective and total violations (if present) from the
                    supplied data.  Requires a string to be used in the iter column
                    and a struct of the data (provided by provided by the
                    makePenaltyFunction object.
                .unscaledMsg
                    Prints an informational message about prescaling, since GRANSO
                    prints the prescaled problem values, except at the end for the
                    results, where it prints both the prescaled and unscaled
                    values.
                .regularizeError
                    Prints a single-line error message to indicate when an eig
                    error causes the regularization procedure to abort.
                .bfgsError
                    Prints a single-line error message corresponding to the integer
                    code given as the single input argument.  For more information 
                    on these codes, see bfgsHessianInverse.
                .qpError
                    Prints a multi-line error message showing the details of a
                    thrown error from attempting to solve a QP.  Requires two
                    arguments: the first is the thrown error, the second is a
                    string to be used as a label to indicate to the user which 
                    piece of code threw this error.       

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
                gransoPrinter.m introduced in GRANSO Version 1.0.

                Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                    ncvxPrinter.py is translated from gransoPrinter.m in GRANSO Version 1.6.4. 

            For comments/bug reports, please visit the NCVX webpage:
            https://github.com/sun-umn/NCVX
                
            NCVX Version 1.0.0, 2021, see AGPL license info below.

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
        self.opts = opts
        self.n = n
        self.n_ineq = n_ineq
        self.n_eq = n_eq

        # Setup printer options from GRANSO options
        ascii                           = self.opts.print_ascii
        use_orange                      = self.opts.print_use_orange
        print_opts = GeneralStruct()
        setattr(print_opts,"use_orange",use_orange)
        setattr(print_opts,"use_ascii",ascii)
        setattr(print_opts,"print_width",self.opts.print_width)
        setattr(print_opts,"maxit",self.opts.maxit)
        setattr(print_opts,"ls_max_estimate",50*(self.opts.linesearch_reattempts + 1))

        [*_,LAST_FALLBACK_LEVEL]         = pC.ncvxConstants()
        if self.opts.max_fallback_level < LAST_FALLBACK_LEVEL:
            setattr(print_opts,"random_attempts",0)
        else:
            setattr(print_opts,"random_attempts",self.opts.max_random_attempts)
        print_opts.ngrad                = self.opts.ngrad + 1

        cols        = pPC.ncvxPrinterColumns(print_opts,self.n_ineq, self.n_eq)
        constrained = self.n_ineq or self.n_eq
        if constrained:
            pen_label           = "Penalty Function"
            viol_label          = "Total Violation"
            self.pen_vals_fn         = lambda penfn_at_x: self.penaltyFunctionValuesConstrained(penfn_at_x)
            if n_ineq and n_eq:
                self.viol_vals_fn    = lambda penfn_at_x: self.violationValuesBoth(penfn_at_x)
            elif n_ineq:
                self.viol_vals_fn    = lambda penfn_at_x: self.violationValuesInequality(penfn_at_x)
            else:
                self.viol_vals_fn    = lambda penfn_at_x: self.violationValuesEquality(penfn_at_x)
            
        else:
            pen_label           = "Penalty Fn"
            viol_label          = "Violation"
            self.pen_vals_fn         = lambda varargin: self.penaltyFunctionValues(varargin)
            self.viol_vals_fn        = lambda varargin: self.violationValues(varargin)


        self.iter_c          = cols.iter
        self.mu_c            = cols.mu
        self.pen_c           = cols.pen
        self.obj_c           = cols.obj
        self.ineq_c          = cols.ineq
        self.eq_c            = cols.eq
        self.sd_c            = cols.sd
        self.ls_evals_c      = cols.ls_evals
        self.ls_size_c       = cols.ls_size
        self.ngrad_c         = cols.ngrad
        self.stat_c          = cols.stat_value
                
        labels = (  self.iter_c.label, self.mu_c.label, self.pen_c.label,
                    self.obj_c.label, self.ineq_c.label, self.eq_c.label,
                    self.sd_c.label,      self.ls_evals_c.label,    self.ls_size_c.label,
                    self.ngrad_c.label,   self.stat_c.label )
                
        widths = (  self.iter_c.width, self.mu_c.width,      self.pen_c.width, self.obj_c.width ,
                    self.ineq_c.width,    self.eq_c.width ,
                    self.sd_c.width,      self.ls_evals_c.width,    self.ls_size_c.width,
                    self.ngrad_c.width,   self.stat_c.width )
                
            
        span_labels     = ((pen_label,2,3),
                            (viol_label,5,6),
                            ('Line Search',7,9),
                            ('Stationarity',10,11))


        tP_obj   = tP()
        self.table_printer = tP_obj.tablePrinter( ascii, use_orange, labels, widths, 1, span_labels )    
        
        print_count     = 0
                
        msg_box_fn = lambda varargin:  pMB.printMessageBox(ascii,use_orange,varargin)
        
        printer = GeneralStruct
        setattr(printer, "msg", lambda s: self.table_printer.msg(s) )
        setattr(printer, "close", lambda : self.table_printer.close() )
        setattr(printer, "msgWidth", lambda : self.table_printer.msgWidth() )
        setattr(printer, "init", lambda penfn_at_x,stat_value,n_qps: self.init(penfn_at_x,stat_value,n_qps))
        setattr(printer, "iter", lambda iter, penfn_at_x,sd_code,random_attempts,ls_evals,alpha,n_grads,stat_value,n_qps:
                                     self.iteration(iter, penfn_at_x,sd_code,random_attempts,ls_evals,alpha,n_grads,stat_value,n_qps))
        setattr(printer, "summary", lambda name,penfn_at_x: self.summary(name,penfn_at_x))
        setattr(printer, "unscaledMsg", lambda : self.unscaledMsg())
        setattr(printer, "lineSearchRestart", lambda iter,mu_ls: self.lineSearchRestart(iter,mu_ls) )
        setattr(printer, "bfgsInfo", lambda iter,code: self.printBfgsInfo(iter,code) )
        setattr(printer, "regularizeError", lambda iter: self. printRegularizeError(iter) )
        setattr(printer, "qpError", lambda iter,err,loc: self.printQPError(iter,err,loc) )
        setattr(printer, "quadprogFailureRate", lambda rate: self.quadprogFailureRate(rate) )
            
        return printer

    def init(self,penfn_at_x,stat_value,n_qps):
        
        self.gransoHeader()
        self.table_printer.header()
        
        self.print_count = 0
        pen_values  = self.pen_vals_fn(penfn_at_x)
        viol_values = self.viol_vals_fn(penfn_at_x)
        info_values = self.infoValues(-1,0,1,0,1)
        
        self.table_printer.row(  (self.iter_c.format_fn(0),) + pen_values + 
                                (self.obj_c.format_fn(penfn_at_x.f),) + viol_values + 
                                info_values + (self.stat_c.format_fn(stat_value,n_qps),) )
        

    def iteration( self,iter, penfn_at_x,sd_code,random_attempts,ls_evals,alpha,n_grads,stat_value,n_qps):
        
        self.print_count = self.print_count + 1
        if self.print_count % 20 == 0:
            self.table_printer.header()
            self.print_count = 0
        
        pen_values  = self.pen_vals_fn(penfn_at_x)
        viol_values = self.viol_vals_fn(penfn_at_x)
        info_values = self.infoValues( sd_code, random_attempts, ls_evals, alpha, n_grads )
                                
        self.table_printer.row(  (self.iter_c.format_fn(iter),) + pen_values + ( self.obj_c.format_fn(penfn_at_x.f),) +
                            viol_values + info_values + (self.stat_c.format_fn(stat_value,n_qps),)          )

    def summary(self,name,penfn_at_x):   
        viol_values = self.viol_vals_fn(penfn_at_x)
        info_values = (self.sd_c.blank_str, self.ls_evals_c.blank_str, self.ls_size_c.blank_str, self.ngrad_c.blank_str)
                        
        self.table_printer.row(  (self.iter_c.format_str(name),) + (self.mu_c.blank_str,)+ 
                            (self.pen_c.blank_str,) + (self.obj_c.format_fn(penfn_at_x.f), ) +
                            viol_values + info_values + (self.stat_c.blank_str,) )
    

    def unscaledMsg(self):
        if self.print_count < 0:
            return
        self.table_printer.overlayOrange(prescalingEndBlockMsg())

    #  column entry preprocessing helper functions 

    def penaltyFunctionValues(self,varargin):
        pen_args = (self.mu_c.dash_str,self.pen_c.dash_str)
        return pen_args

    def penaltyFunctionValuesConstrained(self,penfn_at_x):
        pen_args = (self.mu_c.format_fn(penfn_at_x.mu),self.pen_c.format_fn(penfn_at_x.p.item()))
        return pen_args

    def violationValues(self,varargin):
        viol_args = (self.ineq_c.dash_str,self.eq_c.dash_str)
        return viol_args

    def violationValuesInequality(self,penfn_at_x):
        viol_args = (self.ineq_c.format_fn(penfn_at_x.tvi),self.eq_c.dash_str)
        return viol_args

    def violationValuesEquality(self,penfn_at_x):
        viol_args = (self.ineq_c.dash_str,self.eq_c.format_fn(penfn_at_x.tve) )
        return viol_args

    def violationValuesBoth(self,penfn_at_x):
        viol_args = (self.ineq_c.format_fn(penfn_at_x.tvi),self.eq_c.format_fn(penfn_at_x.tve))
        return viol_args

    def infoValues( self, sd_code, random_attempts, ls_evals, alpha, n_grads ):
                                    
        info_args = (self.sd_c.format_fn(sd_code,random_attempts),self.ls_evals_c.format_fn(ls_evals),self.ls_size_c.format_fn(alpha),self.ngrad_c.format_fn(n_grads))
        return info_args   

    def lineSearchRestart(self,iter,mu_ls):
        msg = ["Line search failed to bracket a minimizer, ",
                "reattempting with mu = %g. (Iter = %d)"%(mu_ls, iter)] 
        self.table_printer.msg(msg); 
    

    def printBfgsInfo(self,iter,code):             
        if code == 1:
            r = "applied but without scaling."
        elif code == 2:
            r = "skipped: sty <= 0."
        elif code == 3:
            r = "skipped: update contained nans/infs."
        else:
            return
        
        self.table_printer.msg( "BFGS update %s (Iter = %d)"%(r,iter) )
    

    def printRegularizeError(self,iter):
        msg =  "Regularization skipped due to eig() error. (Iter = %d)"%(iter) 
        self.table_printer.msgOrange(msg)

    def printQPError(self,iter,err,loc):

        # QPErr NOT used 
        #  title to appear in top/bottom borders
        t1 = "NCVX: QUADPROG ERROR IN %s QP (Iter %d) - START"%(loc,iter)   
        t2 = "NCVX: QUADPROG ERROR IN %s QP - END"%(loc)   
            

    def quadprogFailureRate(self,rate):    
        self.table_printer.msgOrange(quadprogFailureRateMsg(rate))
    

    #  private function to print NCVX's opening header with name, author,
    #  copyright, problem specs, and whether limited-memory mode is active
    def gransoHeader(self):
        copyrightNotice_msg = copyrightNotice()
        self.table_printer.msg(copyrightNotice_msg)
        
        #  print the problem specs
        w       = nDigitsInWholePart(max([self.n, self.n_ineq, self.n_eq])) + 2
        spec_fn = lambda s,n: " %-35s: %*d"%("# of %s"%s,w,n)
        
        msg = ["Problem specifications:", spec_fn("variables",self.n), spec_fn("inequality constraints",self.n_ineq), spec_fn("equality constraints",self.n_eq)]
        self.table_printer.msg(msg)
        
        #  print limited memory message, if enabled
        nvec = self.opts.limited_mem_size
        if nvec <= 0:
            return
        
        msg = ["Limited-memory mode enabled with size = %d."%nvec,
               "NOTE: limited-memory mode is generally NOT ",
               "recommended for nonsmooth problems." ] 

        self.table_printer.msgOrange(msg)
        
        

def prescalingEndBlockMsg():
    s = ["",
        "NCVX applied pre-scaling at x0.  Information:",
        " - ABOVE shows values for the pre-scaled problem",
        " - BELOW shows the unscaled values for the optimization results.",
        "NOTE: the pre-scaled solution MAY NOT be a solution to the original",
        "unscaled problem!  For more details, see opts.prescaling_threshold.",
        ""]

    return s

def quadprogFailureRateMsg(rate):
    s = ["WARNING: NCVX''s performance may have been hindered by issues with QP solver.",
    "quadprog''s failure rate: {}%".format(rate),
    "Ensure that quadprog is working correctly!"]
    return s

