from pyoti.core import get_dHelp

import pyoti.core as coti 

import os

import errno

import re

getpath = coti.whereotilib.getpath

h = get_dHelp()

# Create valid characters for variable naming
# Supports up to 738 bases (0-9, A-Z, a-z, aa-zz)
valid_chars = []

# Create chars '0' to '9' (indices 0-9)
for i in range(48, 58):
    valid_chars.append(chr(i))

# Create chars 'A' to 'Z' (indices 10-35)
for i in range(65, 91):
    valid_chars.append(chr(i))

# Create chars 'a' to 'z' (indices 36-61)
for i in range(97, 123):
    valid_chars.append(chr(i))

# Create two-character combinations for indices 62+ (aa, ab, ..., zz)
for first in range(97, 123):  # a-z
    for second in range(97, 123):  # a-z
        valid_chars.append(chr(first) + chr(second))

endl = "\n"
imdir_base_name = 'e'
operators = ['*', '-', '+', '/', '**']

class writer:

  #---------------------------------------------------------------------------------------------------  

  #***************************************************************************************************
  def __init__(self, nbases, order, tab = "  ", coeff_type = "coeff_t", mdual = False,
    type_name = None ):
    """
    PORPUSE:  The porpuse of this class is to create Modules that allow dense OTI structures
              to be manipulated in languages like Fortran and Cython. 
    """

    global imdir_base_name, h
    
    # Only pre-generated data for up to multidual of 10 imbases.
    if mdual and (nbases <= 10) : 
      self.order   = nbases
    elif not mdual:
      self.order   = order
    else:
      raise ValueError("Multidual with more than 10 bases are not supported.")
    # end if

    self.nbases  = nbases
    self.tab     = tab
    self.coeff_t = coeff_type

    self.get = "."
    self.get_ptr = "->"
    self.comment = "// "
    self.endl = ";\n"
    self.real_str = 'r'
    self.new_line_mark = ''
    self.zero = '0.0'
    imdir_base_name = 'e'
    

    if not mdual:
      
      self.type_name = 'onumm'+str(self.nbases)+"n"+str(self.order)+"_t"
      self.type_name_arr = 'oarrm'+str(self.nbases)+"n"+str(self.order)+"_t"
      self.type_name_fe = 'feonumm'+str(self.nbases)+"n"+str(self.order)+"_t"
      self.type_name_fearr = 'feoarrm'+str(self.nbases)+"n"+str(self.order)+"_t"
      self.func_name = 'onumm'+str(self.nbases)+"n"+str(self.order)
      self.func_name_arr = 'oarrm'+str(self.nbases)+"n"+str(self.order)
      self.func_name_fe = 'feonumm'+str(self.nbases)+"n"+str(self.order)
      self.func_name_fearr = 'feoarrm'+str(self.nbases)+"n"+str(self.order)

      self.pytype_name = 'onumm'+str(self.nbases)+"n"+str(self.order)
      self.pytype_name_arr = 'omatm'+str(self.nbases)+"n"+str(self.order)
      self.pytype_name_fe = 'feonumm'+str(self.nbases)+"n"+str(self.order)
      self.pytype_name_fearr = 'feomatm'+str(self.nbases)+"n"+str(self.order)
      self.pyfunc_name = 'onumm'+str(self.nbases)+"n"+str(self.order)
      self.pyfunc_name_arr = 'omatm'+str(self.nbases)+"n"+str(self.order)
      self.pyfunc_name_fe = 'feomatm'+str(self.nbases)+"n"+str(self.order)
      self.pyfunc_name_fearr = 'feomatm'+str(self.nbases)+"n"+str(self.order)

      self.type_names={}
      self.type_names['r']=self.coeff_t
      self.type_names['o']=self.type_name
      self.type_names['f']=self.type_name_fe
      self.type_names['R']='darr_t'
      self.type_names['O']=self.type_name_arr
      self.type_names['F']=self.type_name_fearr

    else:

      self.type_name = 'mdnum'+str(self.nbases)+"_t"
      self.type_name_arr = 'mdarr'+str(self.nbases)+"_t"
      self.type_name_fe = 'femdnum'+str(self.nbases)+"_t"
      self.type_name_fearr = 'femdarr'+str(self.nbases)+"_t"
      self.func_name = 'mdnum'+str(self.nbases)
      self.func_name_arr = 'mdarr'+str(self.nbases)
      self.func_name_fe = 'femdnum'+str(self.nbases)
      self.func_name_fearr = 'femdarr'+str(self.nbases)

      self.pytype_name = 'mdnum'+str(self.nbases)
      self.pytype_name_arr = 'mdmat'+str(self.nbases)
      self.pytype_name_fe = 'mdnumfe'+str(self.nbases)
      self.pytype_name_fearr = 'mdmatfe'+str(self.nbases)
      self.pyfunc_name = 'mdnum'+str(self.nbases)
      self.pyfunc_name_arr = 'mdmat'+str(self.nbases)
      self.pyfunc_name_fe = 'mdnumfe'+str(self.nbases)
      self.pyfunc_name_fearr = 'mdmatfe'+str(self.nbases)

      self.type_names={}
      self.type_names['r']=self.coeff_t
      self.type_names['o']=self.type_name
      self.type_names['f']=self.type_name_fe
      self.type_names['R']='darr_t'
      self.type_names['O']=self.type_name_arr
      self.type_names['F']=self.type_name_fearr
    # end if 


    self.nimdir = h.get_ndir_total(self.nbases, self.order)
    
    # append imaginary direction.
    self.name_imdir = []
    self.name_imdir.append([])
    self.name_imdir[0].append(self.real_str)

    self.idx_imdir = []
    self.idx_imdir.append([])
    self.idx_imdir[0].append(0)
    
    self.use_imdir = []
    self.use_imdir.append([])
    self.use_imdir[0].append(True)

    # generate all imaginary directions
    for ordi in range(1,self.order+1):

      self.name_imdir.append([])
      self.use_imdir.append([])
      self.idx_imdir.append([])
      
      nimdir_i = h.get_ndir_order(self.nbases, ordi)

      for j in range(nimdir_i):
        
        str_out = imdir_base_name

        list_bases = h.get_fulldir(j,ordi)
        
        for i in range(list_bases.size):

          str_out += valid_chars[list_bases[i]]

        # end for        

        
        if not mdual:
          self.name_imdir[ordi].append(str_out)
          self.use_imdir[ordi].append(True)
          self.idx_imdir[ordi].append(j)
        else:
          # Check if it is a valid multidual direction.
          str_test = "".join(dict.fromkeys(str_out))
          if str_test == str_out:
            self.name_imdir[ordi].append(str_out)
            self.use_imdir[ordi].append(True)
            self.idx_imdir[ordi].append(j)
          # end if 
      # end for 
    # end for 

    # Precompute multiplication
    self.mult_res = []
    self.mult_res.append([]) # Order 0.
    self.mult_res_total = []
    self.mult_res_total.append([[[[0,0,self.real_str],[0,0,self.real_str]]]]) # Order 0.

    for ordi in range(1,self.order+1):

      dirs = self.name_imdir[ordi]
      idxi = self.idx_imdir[ordi]

      self.mult_res.append([])
      mults = self.mult_res[-1]

      for j in range(len(dirs)):
        mults.append([]) 
      # end for      

      for ordj in range(1, ordi // 2 + 1):
        
        ordk = ordi - ordj

        dirsj = self.name_imdir[ordj]
        dirsk = self.name_imdir[ordk]

        idxj = self.idx_imdir[ordj]
        idxk = self.idx_imdir[ordk]

        for j in range(len(dirsj)):
          for k in range(len(dirsk)):

            i,iordi = h.mult_dir(idxj[j],ordj,idxk[k],ordk)

            if i in idxi:
              ii = idxi.index(i)
              mults[ii].append([ dirsj[j], dirsk[k] ])
              if  ordj != ordk:
                mults[ii].append([ dirsk[k],dirsj[j] ])
              # end if 

            #end if 
            
          # end for
        # end for 
      # end for 
    # end for 

    for ordi in range(1,self.order+1):

      dirs = self.name_imdir[ordi]
      idxi = self.idx_imdir[ordi]

      self.mult_res_total.append([])
      mults_total = self.mult_res_total[-1]

      for j in range(len(dirs)):
        mults_total.append([]) 
      # end for
      
      # Do here ordj = 0
      ordj = 0; j=0;
      ordk = ordi - ordj
      dirsj = self.name_imdir[ordj]
      dirsk = self.name_imdir[ordk]

      idxj = self.idx_imdir[ordj]
      idxk = self.idx_imdir[ordk]
      for k in range(len(dirsk)):

        i,iordi = (idxk[k],ordk)

        if i in idxi:
          ii = idxi.index(i)
          mults_total[ii].append([ [0,ordj,dirsj[j]], [idxk[k],ordk,dirsk[k]] ])
          if  ordj != ordk:
            mults_total[ii].append([ [idxk[k],ordk,dirsk[k]], [idxj[j],ordj,dirsj[j]] ])
          # end if 
        #end if 
        
      # end for

      for ordj in range(1, ordi // 2 + 1):
        
        ordk = ordi - ordj

        dirsj = self.name_imdir[ordj]
        dirsk = self.name_imdir[ordk]

        idxj = self.idx_imdir[ordj]
        idxk = self.idx_imdir[ordk]

        for j in range(len(dirsj)):
          for k in range(len(dirsk)):

            i,iordi = h.mult_dir(idxj[j],ordj,idxk[k],ordk)

            if i in idxi:
              ii = idxi.index(i)
              mults_total[ii].append([ [idxj[j],ordj,dirsj[j]], [idxk[k],ordk,dirsk[k]] ])
              if  ordj != ordk:
                mults_total[ii].append([ [idxk[k],ordk,dirsk[k]], [idxj[j],ordj,dirsj[j]] ])
              # end if 

            # #end if 
            
          # end for
        # end for 
      # end for 
    # end for 

    self.function_list = []
    self.function_list_header = {}

    self.function_list_header['base'] = []
    self.function_list_header['algebra'] = []

    # needed for c?
    self.overloads = {}

    self.overloads['*'] = []
    self.overloads['+'] = []
    self.overloads['-'] = []
    self.overloads['/'] = []
    self.overloads['='] = []
    self.overloads['pow'] = []
    self.overloads['print'] = []
    self.overloads['transpose'] = []
    self.overloads['matmul'] = []
    self.overloads['sin']   = []
    self.overloads['cos']   = []
    self.overloads['tan']   = []
    self.overloads['asin']  = []
    self.overloads['acos']  = []
    self.overloads['atan']  = []
    self.overloads['sinh']  = []
    self.overloads['cosh']  = []
    self.overloads['tanh']  = []
    self.overloads['asinh'] = []
    self.overloads['acosh'] = []
    self.overloads['atanh'] = []
    self.overloads['log']   = []
    self.overloads['exp']   = []
    self.overloads['log10'] = []


  #---------------------------------------------------------------------------------------------------  

  #***************************************************************************************************
  def set_type_c(self, level = ""):
    """
    PORPUSE:  The porpuse of this class is to create Modules that allow dense OTI structures
              to be manipulated in languages like Fortran C, C++ and Cython. 
    """
    global h
    str_out = ""

    # -------------------- Scalar
    
    str_out += level + self.comment + "Scalar" + endl
    
    str_out += level + "typedef struct "+"{"+endl
    # Write real part.
    str_out += level + self.tab + self.comment + "Real" + endl
    str_out += level + self.tab + self.coeff_t + " " + self.real_str + self.endl

    for ordi in range(1,self.order+1):

      str_out += level + self.tab + self.comment + "Order " + str(ordi) + endl
      dirs = self.name_imdir[ordi]

      for j in range(len(dirs)):

        str_out += level + self.tab + self.coeff_t + " " + dirs[j] + self.endl

      # end for 

    # end for 

    str_out += level + "} " + self.type_name + self.endl + endl

    # # -------------------- Array
    # str_out += level + self.comment + "Array" + endl
    
    # str_out += level + "typedef struct "+"{"+endl
    # str_out += level + self.tab + self.comment + "Data" + endl
    # str_out += level + self.tab + self.type_name + "* p_data" + self.endl
    # str_out += level + self.tab + "uint64_t nrows" + self.endl
    # str_out += level + self.tab + "uint64_t ncols" + self.endl
    # str_out += level + self.tab + "uint64_t size " + self.endl
    # str_out += level + "} " + self.type_name_arr + self.endl + endl

    # # -------------------- Gauss Structures
    # str_out += level + self.comment + "Gauss Scalar" + endl
    # str_out += level + "typedef struct "+"{"+endl
    # str_out += level + self.tab + self.comment + "Data" + endl
    # str_out += level + self.tab + self.type_name + "* p_data" + self.endl
    # str_out += level + self.tab + "uint64_t nip" + self.endl
    # str_out += level + "} " + self.type_name_fe + self.endl + endl
    

    # # -------------------- Gauss Array
    # str_out += level + self.comment + "Gauss Array" + endl
    # str_out += level + "typedef struct "+"{"+endl
    # str_out += level + self.tab + self.comment + "Data" + endl
    # str_out += level + self.tab + self.type_name_arr + "* p_data" + self.endl
    # str_out += level + self.tab + "uint64_t nrows" + self.endl
    # str_out += level + self.tab + "uint64_t ncols" + self.endl
    # str_out += level + self.tab + "uint64_t size " + self.endl
    # str_out += level + self.tab + "uint64_t nip  " + self.endl
    # str_out += level + "} " +  self.type_name_fearr + self.endl + endl

    return str_out

  #---------------------------------------------------------------------------------------------------  
  
  #***************************************************************************************************
  def multiplication_like_function_ro(self, level = "", f_name = "FUNCTION", lhs_name= "LHS", lhs_ptr=False,
    rhs_name= "RHS", rhs_ptr=False, res_name = "RES", separator = ",", f_open = "(", f_close = ")", 
    addition = " + ", to=False  ):
    """
    PORPUSE:  multiplication like operation between real and OTIs. (LHS is REAL and RHS is OTI)
    """
    global h

    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if

    if rhs_ptr:
      rhs_getter = self.get_ptr
    else:
      rhs_getter = self.get
    # end if
    
    str_out = ""

    str_out += level + self.comment + "Multiplication like function \'"
    str_out += f_name + f_open + lhs_name + separator + rhs_name + f_close
    str_out += "\'\n"


    # Write real part.
    str_out += level +  self.comment + " Real" + self.endl
    str_out += level + res_name + res_getter + self.real_str + " = "
    str_out += f_name + f_open
    str_out +=         lhs_name + separator
    str_out +=         rhs_name + rhs_getter + self.real_str + f_close + self.endl

    for ordi in range(1,self.order+1):

      str_out += level + self.comment + "Order " + str(ordi) + self.endl
      dirs = self.name_imdir[ordi]

      for j in range(len(dirs)):

        str_out += level  + res_name + res_getter + dirs[j] + " = "
        str_out += f_name + f_open
        str_out +=         lhs_name + separator
        str_out +=         rhs_name + rhs_getter + dirs[j] + f_close + self.endl
        
      # end for 
    # end for 

    return str_out

  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def multiplication_like_function_or(self, level = "", f_name = "FUNCTION", lhs_name= "LHS", lhs_ptr=False,
    rhs_name= "RHS", rhs_ptr=False, res_name = "RES", separator = ",", f_open = "(", f_close = ")", 
    addition = " + ", to=False  ):
    """
    PORPUSE:  multiplication like operation between real and OTIs. (LHS is REAL and RHS is OTI)
    """
    global h

    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if

    if rhs_ptr:
      rhs_getter = self.get_ptr
    else:
      rhs_getter = self.get
    # end if
    
    str_out = ""

    str_out += level + self.comment + "Multiplication like function \'"
    str_out += f_name + f_open + lhs_name + separator + rhs_name + f_close
    str_out += "\'" + endl


    # Write real part.
    str_out += level + self.comment + " Real" + self.endl
    str_out += level + res_name + res_getter + self.real_str + " = "
    str_out += f_name + f_open
    str_out +=         lhs_name + lhs_getter + self.real_str + separator
    str_out +=         rhs_name + f_close + self.endl

    for ordi in range(1,self.order+1):

      str_out += level +self.comment + "Order " + str(ordi) + self.endl
      dirs = self.name_imdir[ordi]

      for j in range(len(dirs)):

        str_out += level + res_name + res_getter + dirs[j] + " = "
        str_out += f_name + f_open
        str_out +=         lhs_name + lhs_getter + dirs[j] + separator
        str_out +=         rhs_name + f_close + self.endl
        
      # end for 

    # end for 

    return str_out

  #---------------------------------------------------------------------------------------------------  

  #***************************************************************************************************
  def multiplication_like_function_oo(self, level = "", f_name = "FUNCTION", lhs_name= "LHS", lhs_ptr=False,
    rhs_name= "RHS", rhs_ptr=False, res_name = "RES", separator = ",", f_open = "(", f_close = ")", 
    addition = " + ", to=False  ):
    """
    PORPUSE:  Multiplication like operation between OTI and OTI.
    """
    global h
    
    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if

    if rhs_ptr:
      rhs_getter = self.get_ptr
    else:
      rhs_getter = self.get
    # end if

    str_out = ""

    str_out += level + self.comment + " Multiplication like function \'"
    str_out += f_name + f_open + lhs_name + separator + rhs_name + f_close
    str_out += "\'\n"


    # Write real part.
    str_out += level + self.comment + "Real" + self.endl
    str_out += level + res_name + res_getter + self.real_str + " = "
    str_out += f_name + f_open
    str_out +=         lhs_name + lhs_getter + self.real_str + separator
    str_out +=         rhs_name + rhs_getter + self.real_str + f_close + self.endl

    # res = []

    for ordi in range(1,self.order+1):

      str_out += level +self.comment + "Order " + str(ordi) + self.endl
      dirs = self.name_imdir[ordi]
      idxi = self.idx_imdir[ordi]

      mults = []
      for j in range(len(dirs)):
        mults.append([]) 

      # Multiply the different imaginary directions all togeather such that resulting order is 
      # ordi.
      # print("Order "+str(ordi))

      for ordj in range(1, ordi // 2 + 1):
        ordk = ordi - ordj
        # print("  Multiplying order "+str(ordj)+" x order " + str(ordk) )
        dirsj = self.name_imdir[ordj]
        dirsk = self.name_imdir[ordk]

        idxj = self.idx_imdir[ordj]
        idxk = self.idx_imdir[ordk]

        for j in range(len(dirsj)):
          for k in range(len(dirsk)):

            i,iordi = h.mult_dir(idxj[j],ordj,idxk[k],ordk)

            if i in idxi:
              ii = idxi.index(i)
              mults[ii].append([ dirsj[j], dirsk[k] ])
              if  ordj != ordk:
                mults[ii].append([ dirsk[k],dirsj[j] ])
              # end if 
            #end if 
          # end for
        # end for 
      # end for 


      for j in range(len(dirs)):
        # R X IM
        str_out += level + res_name + res_getter + dirs[j] + " = " 
        str_out += f_name + f_open
        str_out +=         lhs_name + lhs_getter + self.real_str + separator
        str_out +=         rhs_name + rhs_getter + dirs[j] 
        str_out += f_close 

        # IM X R
        # Addition
        str_out += addition
        str_out += f_name + f_open
        str_out +=         lhs_name + lhs_getter + dirs[j] + separator
        str_out +=         rhs_name + rhs_getter + self.real_str 
        str_out += f_close 

        mult = mults[j]
        nterms = len(mult)
        for k in range(nterms):

          str_out += addition
          str_out += f_name + f_open
          str_out +=         lhs_name + lhs_getter + mult[k][0] + separator
          str_out +=         rhs_name + rhs_getter + mult[k][1]
          str_out += f_close 

          if ( (k+3)%3 == 0 ) and ( k != ( nterms-1 ) ) :
            str_out += " " + self.new_line_mark + endl
            str_out += level +' '*(len(res_name)+len(dirs[j])+1)
          # end if 


        # end for 
        str_out += self.endl


        
      # end for 

    # end for 

    return str_out

  #---------------------------------------------------------------------------------------------------  

  #***************************************************************************************************
  def truncmul_like_function_oo(self, level = "", f_name = "FUNCTION", lhs_name= "LHS", lhs_ptr=False,
    rhs_name= "RHS", rhs_ptr=False, res_name = "RES", separator = ",", f_open = "(", f_close = ")", 
    addition = " + ", to=False  ):
    """
    PORPUSE:  Multiplication like operation between OTI and OTI.
    """
    global h
    
    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if

    if rhs_ptr:
      rhs_getter = self.get_ptr
    else:
      rhs_getter = self.get
    # end if

    str_out = ""

    if not to:

      str_out += level + res_name+" = "+self.func_name+"_init()" + self.endl
      str_out += endl

    else:

      str_out += level + " (*" + res_name+") = "+self.func_name+"_init()" + self.endl
      str_out += endl

    
    # end if 
    str_out += level + self.comment + " Truncated multiplication like function \'"
    str_out += f_name + f_open + lhs_name + separator + rhs_name + f_close
    str_out += "\'\n"

    for ordi in range(2,self.order+1):

      str_out += level +self.comment + "Order " + str(ordi) + self.endl
      dirs = self.name_imdir[ordi]
      idxi = self.idx_imdir[ordi]

      mults = []
      for j in range(len(dirs)):
        mults.append([]) 

      # Multiply the different imaginary directions all togeather such that resulting order is 
      # ordi.
      # print("Order "+str(ordi))

      for ordj in range(1, ordi // 2 + 1):
        ordk = ordi - ordj
        # print("  Multiplying order "+str(ordj)+" x order " + str(ordk) )
        dirsj = self.name_imdir[ordj]
        dirsk = self.name_imdir[ordk]

        idxj = self.idx_imdir[ordj]
        idxk = self.idx_imdir[ordk]
        
        for j in range(len(dirsj)):
          for k in range(len(dirsk)):

            i,iordi = h.mult_dir(idxj[j],ordj,idxk[k],ordk)

            if i in idxi:
              ii = idxi.index(i)
              mults[ii].append([ dirsj[j], dirsk[k] ])
              if  ordj != ordk:
                mults[ii].append([ dirsk[k],dirsj[j] ])
              # end if 
            #end if 
          # end for
        # end for 
      # end for 

      # res.append(mults)

      for j in range(len(dirs)):
        # # R X IM
        str_out += level + res_name + res_getter + dirs[j] + " = " 
        

        mult = mults[j]
        nterms = len(mult)
        for k in range(nterms):

          str_out += addition
          str_out += f_name + f_open
          str_out +=         lhs_name + lhs_getter + mult[k][0] + separator
          str_out +=         rhs_name + rhs_getter + mult[k][1]
          str_out += f_close 

          if ( (k+1)%3 == 0 ) and ( k != ( nterms-1 ) ) :
            str_out += " " + self.new_line_mark + endl
            str_out += level +' '*(len(res_name)+len(dirs[j])+1)
          # end if 


        # end for 
        str_out += self.endl


        
      # end for 

    # end for 

    return str_out

  #---------------------------------------------------------------------------------------------------  


  #***************************************************************************************************
  def gem_like_function_oo(self, level = "", f_name = "FUNCTION", a_name= "a", a_ptr=False,
    b_name= "b", b_ptr=False, c_name= "c", c_ptr=False, res_name = "res", separator = ",", 
    f_open = "(", f_close = ")",  addition = " + ", to=False  ):
    """
    PORPUSE:  Multiplication like operation between OTI and OTI.
    """
    global h

    
    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if a_ptr:
      a_getter = self.get_ptr
    else:
      a_getter = self.get
    # end if

    if b_ptr:
      b_getter = self.get_ptr
    else:
      b_getter = self.get
    # end if

    if c_ptr:
      c_getter = self.get_ptr
    else:
      c_getter = self.get
    # end if


    str_out = ""

    str_out += level + self.comment + " General multiplication like function \'"
    str_out += f_name + f_open + a_name + separator + b_name + f_close
    str_out += addition + c_name
    str_out += "\'\n"


    # Write real part.
    str_out += level + self.comment + "Real" + self.endl
    str_out += level + res_name + res_getter + self.real_str + " = "
    str_out +=         c_name + c_getter + self.real_str + addition
    str_out += f_name + f_open
    str_out +=         a_name + a_getter + self.real_str + separator
    str_out +=         b_name + b_getter + self.real_str + f_close + self.endl

    # res = []

    for ordi in range(1,self.order+1):

      str_out += level +self.comment + "Order " + str(ordi) + self.endl
      dirs = self.name_imdir[ordi]
      idxi = self.idx_imdir[ordi]

      mults = []
      for j in range(len(dirs)):
        mults.append([]) 

      # Multiply the different imaginary directions all togeather such that resulting order is 
      # ordi.
      # print("Order "+str(ordi))

      for ordj in range(1, ordi // 2 + 1):
        ordk = ordi - ordj
        # print("  Multiplying order "+str(ordj)+" x order " + str(ordk) )
        dirsj = self.name_imdir[ordj]
        dirsk = self.name_imdir[ordk]

        idxj = self.idx_imdir[ordj]
        idxk = self.idx_imdir[ordk]
        
        for j in range(len(dirsj)):
          for k in range(len(dirsk)):

            i,iordi = h.mult_dir(idxj[j],ordj,idxk[k],ordk)

            if i in idxi:
              ii = idxi.index(i)
              mults[ii].append([ dirsj[j], dirsk[k] ])
              if  ordj != ordk:
                mults[ii].append([ dirsk[k],dirsj[j] ])
              # end if 
            #end if 
          # end for
        # end for 
      # end for 

      # res.append(mults)

      for j in range(len(dirs)):
        # R X IM
        str_out += level + res_name + res_getter + dirs[j] + " = " 
        str_out +=         c_name + c_getter + dirs[j] + addition
        str_out += f_name + f_open
        str_out +=         a_name + a_getter + self.real_str + separator
        str_out +=         b_name + b_getter + dirs[j] 
        str_out += f_close 

        # IM X R
        # Addition
        str_out += addition
        str_out += f_name + f_open
        str_out +=         a_name + a_getter + dirs[j] + separator
        str_out +=         b_name + b_getter + self.real_str 
        str_out += f_close 

        mult = mults[j]
        nterms = len(mult)
        for k in range(nterms):

          str_out += addition
          str_out += f_name + f_open
          str_out +=         a_name + a_getter + mult[k][0] + separator
          str_out +=         b_name + b_getter + mult[k][1]
          str_out += f_close 

          if ( (k+3)%3 == 0 ) and ( k != ( nterms-1 ) ) :
            str_out += " " + self.new_line_mark + endl
            str_out += level +' '*(len(res_name)+len(dirs[j])+1)
          # end if 


        # end for 
        str_out += self.endl


        
      # end for 

    # end for 

    return str_out

  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def truncgem_like_function_oo(self, level = "", f_name = "FUNCTION", a_name= "a", a_ptr=False,
    b_name= "b", b_ptr=False, c_name= "c", c_ptr=False, res_name = "res", separator = ",", 
    f_open = "(", f_close = ")",  addition = " + ", to=False, tab=" "  ):
    """
    PORPUSE:  Truncated General multiplication  like operation between OTI and OTI.
    """
    global h

    
    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if a_ptr:
      a_getter = self.get_ptr
    else:
      a_getter = self.get
    # end if

    if b_ptr:
      b_getter = self.get_ptr
    else:
      b_getter = self.get
    # end if

    if c_ptr:
      c_getter = self.get_ptr
    else:
      c_getter = self.get
    # end if


    str_out = ""


    str_out += level + self.comment + " General multiplication like function \'"
    str_out += f_name + f_open + a_name + separator + b_name + f_close
    str_out += addition + c_name
    str_out += "\'\n"


    str_out += level + "switch( ord_lhs ){" + endl
    leveli = level + tab

    for ord_lhs in range(self.order+1):
      leveli = level + tab
      str_out += leveli + "case "+str(ord_lhs) + ":"+endl 
      
      leveli = level + 2*tab
      str_out += leveli + "switch( ord_rhs ){"+endl       

      for ord_rhs in range(self.order-ord_lhs+1):
        
        ord_res = ord_lhs + ord_rhs
        
        leveli = level + 3*tab
        str_out += leveli + "case "+str(ord_rhs) + ":"+endl 
        
        leveli = level + 4*tab
        str_out += leveli + self.comment + "res order: " + str(ord_res) + endl

        for i in range(len(self.mult_res_total[ord_res])):
          
          mults   = self.mult_res_total[ord_res][i]
          res_dir = self.name_imdir[ord_res][i]

          str_out += leveli + res_name + res_getter + res_dir + ' = '
          str_out += c_name + c_getter + res_dir
          k=1
          
          for mult in mults:

            lhs = mult[0]
            rhs = mult[1]

            if lhs[1]==ord_lhs and rhs[1]==ord_rhs:
              
              if ( (k+3)%3 == 0 ):
                str_out += " " + self.new_line_mark + endl
                str_out += leveli +' '*( len(res_name+res_getter+res_dir) )
              # end if 

              str_out += addition
              str_out += f_name + f_open
              str_out +=         a_name + a_getter + lhs[2] + separator
              str_out +=         b_name + b_getter + rhs[2]
              str_out += f_close 

              k+=1

            # end if 
            
          # end for
          str_out += self.endl
        # end for 
        str_out += leveli + "break"+self.endl
        
      # end for 
      # str_out += leveli + "break"+self.endl
      leveli = level + 2*tab
      str_out += leveli + "}"+endl 
      str_out += leveli + "break"+self.endl 
    # end for 
    str_out += level + "}" + endl
    return str_out

  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def gem_like_function_ro(self, level = "", f_name = "FUNCTION", a_name= "a", a_ptr=False,
    b_name= "b", b_ptr=False, c_name= "c", c_ptr=False, res_name = "res", separator = ",", 
    f_open = "(", f_close = ")",  addition = " + ", to=False  ):
    """
    PORPUSE:  Multiplication like operation between OTI and OTI.
    """
    global h

    
    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if a_ptr:
      a_getter = self.get_ptr
    else:
      a_getter = self.get
    # end if

    if b_ptr:
      b_getter = self.get_ptr
    else:
      b_getter = self.get
    # end if

    if c_ptr:
      c_getter = self.get_ptr
    else:
      c_getter = self.get
    # end if


    str_out = ""

    str_out += level + self.comment + " General multiplication like function \'"
    str_out += f_name + f_open + a_name + separator + b_name + f_close
    str_out += addition + c_name
    str_out += "\'\n"


    # Write real part.
    str_out += level + self.comment + "Real" + self.endl
    str_out += level + res_name + res_getter + self.real_str + " = "
    str_out +=         c_name + c_getter + self.real_str + addition
    str_out += f_name + f_open
    str_out +=         a_name + separator
    str_out +=         b_name + b_getter + self.real_str + f_close + self.endl

    # res = []

    for ordi in range(1,self.order+1):

      str_out += level +self.comment + "Order " + str(ordi) + self.endl
      dirs = self.name_imdir[ordi]

      for j in range(len(dirs)):

        str_out += level + res_name + res_getter + dirs[j] + " = "
        str_out +=         c_name + c_getter + dirs[j] + addition
        str_out += f_name + f_open
        str_out +=         a_name + separator
        str_out +=         b_name + b_getter + dirs[j] + f_close + self.endl
        
      # end for 

    # end for 

    return str_out

  #---------------------------------------------------------------------------------------------------



  #***************************************************************************************************
  def addition_like_function_oo(self, level = "", f_name = "FUNCTION", lhs_name= "LHS", lhs_ptr=False,
    rhs_name= "RHS", rhs_ptr=False, res_name = "RES", separator = ",", f_open = "(", f_close = ")", 
    addition = " + ", to=False  ):
    """
    PORPUSE:  Addition like function between two OTIs.
    """
    global h
    str_out = ""

    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if

    if rhs_ptr:
      rhs_getter = self.get_ptr
    else:
      rhs_getter = self.get
    # end if
    
    str_out += level + self.comment + "Addition like function \'"
    str_out += f_name + f_open + lhs_name + separator + rhs_name + f_close
    str_out += "\'"+ endl


    # Write real part.
    str_out += level + self.comment + " Real" + endl
    str_out += level + res_name + res_getter + self.real_str + " = "
    str_out += f_name + f_open
    str_out +=         lhs_name + lhs_getter + self.real_str + separator
    str_out +=         rhs_name + rhs_getter + self.real_str + f_close + self.endl


    for ordi in range(1,self.order+1):
      
      str_out += level +self.comment + "Order " + str(ordi) + endl
      dirs = self.name_imdir[ordi]

      for j in range(len(dirs)):
        
        str_out += level + res_name + res_getter + dirs[j] + " = "
        str_out += f_name + f_open
        str_out +=         lhs_name + lhs_getter + dirs[j] + separator
        str_out +=         rhs_name + rhs_getter + dirs[j] + f_close + self.endl
        
      # end for 

    # end for 

    return str_out

  #---------------------------------------------------------------------------------------------------  

  #***************************************************************************************************
  def truncaddition_like_function_oo(self, level = "", f_name = "FUNCTION", lhs_name= "LHS", lhs_ptr=False,
    rhs_name= "RHS", rhs_ptr=False, res_name = "RES", separator = ",", f_open = "(", f_close = ")", tab = " ",
    addition = " + ", to=False  ):
    """
    PORPUSE:  Addition like function between two OTIs.
    """
    global h
    str_out = ""

    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if

    if rhs_ptr:
      rhs_getter = self.get_ptr
    else:
      rhs_getter = self.get
    # end if
    
    str_out += level + self.comment + "Addition like function \'"
    str_out += f_name + f_open + lhs_name + separator + rhs_name + f_close
    str_out += "\'"+ endl


    # Write real part.
    str_out += level + "switch(order){" + endl
    leveli = level + tab

    str_out += leveli + "case(0):" + endl
    
    leveli = level + 2*tab

    str_out += leveli + self.comment + " Real" + endl
    str_out += leveli + res_name + res_getter + self.real_str + " = "
    str_out += f_name + f_open
    str_out +=         lhs_name + lhs_getter + self.real_str + separator
    str_out +=         rhs_name + rhs_getter + self.real_str + f_close + self.endl
    str_out += leveli +"break"+ self.endl

    for ordi in range(1,self.order+1):
      leveli = level + tab
      
      str_out += leveli + "case("+str(ordi)+"):"
      str_out += leveli +self.comment + "Order " + str(ordi) + endl
      
      dirs = self.name_imdir[ordi]

      leveli = level + 2*tab
      
      for j in range(len(dirs)):
        
        str_out += leveli + res_name + res_getter + dirs[j] + " = "
        str_out += f_name + f_open
        str_out +=         lhs_name + lhs_getter + dirs[j] + separator
        str_out +=         rhs_name + rhs_getter + dirs[j] + f_close + self.endl
        
      # end for 

      str_out += leveli +"break"+ self.endl

    # end for 

    str_out += level + "}"

    return str_out

  #---------------------------------------------------------------------------------------------------  


  #***************************************************************************************************
  def feval_function_o(self, level = "", lhs_name= "x", lhs_ptr=True,
    feval_name= "feval_re", res_name = "res",  to=False  ):
    """
    PORPUSE:  Generic function evaluation of OTI number.
    """
    global h
    str_out = ""

    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if
    
    str_out += level + self.comment + "feval function" +  endl

    # Required variables 
    str_out += level + self.comment + " Definitions" + endl
    str_out += level + self.coeff_t + " factor=1, coef = 0"+self.endl
    str_out += level + self.type_name + " deltax = (*"+lhs_name+"), deltax_power = (*"+lhs_name+")"+self.endl
    str_out += level +"deltax.r = " + self.zero + self.endl
    str_out += level +"deltax_power.r = " + self.zero + self.endl

    str_out += level + self.func_name + "_set_r("+self.zero+", "
    
    if not to:
      str_out+="&"
    # end if 
    str_out += res_name
    str_out += ")"+self.endl
    
    # Write real part.
    str_out += level + self.comment + " Real" + endl
    str_out += level + res_name + res_getter + self.real_str + " = "
    str_out += feval_name+"[0]" + self.endl
    
    for ordi in range(1,self.order+1):
      
      str_out += level + self.comment + "Order " + str(ordi) + endl
      dirs = self.name_imdir[ordi]

      str_out += level + "factor *="+str(ordi)+self.endl
      str_out += level + "coef    = "+feval_name+"["+str(ordi)+"]/ factor;"+self.endl
      
      str_out += level 
      
      str_out += self.func_name + "_gem_ro_to( coef, &deltax_power, "
      if not to:
        str_out+="&"
      # end if  
      str_out += res_name + ', '
      if not to:
        str_out+="&"
      # end if  
      str_out += res_name  +  ")" + self.endl

      if ordi != self.order:
        str_out += level +"deltax_power = " + self.func_name + "_mul_oo(&deltax_power,&deltax)" + self.endl
      # end if 

    # end for 

    return str_out

  #---------------------------------------------------------------------------------------------------  

  #***************************************************************************************************
  def addition_like_function_ro(self, level = "", f_name = "FUNCTION", lhs_name= "LHS", lhs_ptr=False,
    rhs_name= "RHS", rhs_ptr=False, res_name = "RES", separator = ",", f_open = "(", f_close = ")", 
    addition = " + ", to=False  ):
    """
    PORPUSE:  Addition like function between two OTIs.
    """
    global h
    str_out = ""

    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if

    if rhs_ptr:
      rhs_getter = self.get_ptr
    else:
      rhs_getter = self.get
    # end if

    str_out += level + self.comment + "Addition like function \'"
    str_out += f_name + f_open + lhs_name + separator + rhs_name + f_close
    str_out += "\'\n"


    # Write real part.
    str_out += level + self.comment + "Real" + self.endl
    str_out += level + res_name + res_getter + self.real_str + " = "
    str_out += f_name + f_open
    str_out +=         lhs_name + separator
    str_out +=         rhs_name + rhs_getter + self.real_str + f_close + self.endl


    for ordi in range(1,self.order+1):
      
      str_out += level +self.comment + "Order " + str(ordi) + self.endl
      dirs = self.name_imdir[ordi]

      for j in range(len(dirs)):
        
        str_out += level + res_name + res_getter + dirs[j] + " = "
        str_out += separator + rhs_name + rhs_getter + dirs[j] + self.endl
        
      # end for 

    # end for 

    return str_out

  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def addition_like_function_or(self, level = "", f_name = "FUNCTION", lhs_name= "LHS", lhs_ptr=False,
    rhs_name= "RHS", rhs_ptr=False, res_name = "RES", separator = ",", f_open = "(", f_close = ")", 
    addition = " + ", to=False  ):
    """
    PORPUSE:  Addition like function between two OTIs.
    """
    global h
    str_out = ""

    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if

    if rhs_ptr:
      rhs_getter = self.get_ptr
    else:
      rhs_getter = self.get
    # end if

    str_out += level + self.comment + "Addition like function \'"
    str_out += f_name + f_open + lhs_name + separator + rhs_name + f_close
    str_out += "\'\n"


    # Write real part.
    str_out += level + self.comment + "Real" + self.endl
    str_out += level + res_name + res_getter + self.real_str + " = "
    str_out += f_name + f_open
    str_out +=         lhs_name + lhs_getter + self.real_str + separator
    str_out +=         rhs_name + f_close + self.endl


    for ordi in range(1,self.order+1):
      
      str_out += level +self.comment + "Order " + str(ordi) + self.endl
      dirs = self.name_imdir[ordi]

      for j in range(len(dirs)):
        
        str_out += level + res_name + res_getter + dirs[j] + " = "
        str_out +=         lhs_name + lhs_getter + dirs[j] + self.endl
        
      # end for 

    # end for 

    return str_out

  #--------------------------------------------------------------------------------------------------- 


  #***************************************************************************************************
  def assignr_like_function(self, level = "", f_name = "FUNCTION", lhs_name= "LHS",lhs_ptr=True,
    res_name = "RES", f_open = "(", f_close = ")", to=False):
    """
    PORPUSE:  Addition like function between two OTIs.
    """
    global h
    str_out = ""

    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if 

    str_out += level + self.comment + "Assign like function \'"
    str_out += f_name + f_open + lhs_name + f_close
    str_out += "\'\n"


    # Write real part.
    str_out += level + self.comment + "Real" + self.endl
    str_out += level + res_name + res_getter + self.real_str + " = "
    str_out += f_name + f_open
    str_out +=         lhs_name + f_close + self.endl


    for ordi in range(1,self.order+1):
      
      str_out += level +self.comment + "Order " + str(ordi) + self.endl
      dirs = self.name_imdir[ordi]

      for j in range(len(dirs)):
        
        str_out += level + res_name + res_getter + dirs[j] + " = "+ self.zero + self.endl
        
      # end for 

    # end for 

    return str_out
  #--------------------------------------------------------------------------------------------------- 


  #***************************************************************************************************
  def taylorint_like_function(self, level = "", f_name = "FUNCTION", lhs_name= "LHS",lhs_ptr=True,
    res_name = "RES", f_open = "(", f_close = ")", to=False):
    """
    PORPUSE:  Addition like function between two OTIs.
    """
    global h
    str_out = ""

    if to:
      res_getter = self.get_ptr
      res_prev = "*"
    else:
      res_getter = self.get
      res_prev = ""
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if 

    str_out += level + res_prev + res_name + " = " + self.func_name + "_init()" + self.endl
    str_out += level + self.coeff_t + ' factor' + self.endl
    str_out += level + self.comment + "Taylor integration"+endl

    # Write real part.
    str_out += level + self.comment + "Real" + self.endl
    str_out += level + res_name + res_getter + self.real_str + " = "
    str_out += f_name + f_open
    str_out +=         lhs_name + lhs_getter + self.real_str+ f_close + self.endl


    for ordi in range(1,self.order+1):
      
      str_out += level +self.comment + "Order " + str(ordi) + self.endl
      dirs = self.name_imdir[ordi]

      for j in range(len(dirs)):
        
        str_out += level + 'factor = ' 
        
        for k in range(self.nbases):
          expon = dirs[j].count(str(k+1))
          str_out += 'pow(deltas['+ str(k) + '],'+str(expon)+')*'
        # end for 

        str_out = str_out[:-1]

        str_out += self.endl
        
        str_out += level + res_name + res_getter + self.real_str + " += factor*"
        str_out += f_name + f_open
        str_out +=         lhs_name + lhs_getter + dirs[j] + f_close + self.endl 

      # end for 

    # end for 

    return str_out
  #---------------------------------------------------------------------------------------------------

  #***************************************************************************************************
  def init_like_function(self, level = "", res_name = "RES"):
    """
    PORPUSE:  Addition like function between two OTIs.
    """
    global h
    res_getter = self.get
    str_out = ""
    str_out += level + self.comment + "Init function\n"

    # Write real part.
    str_out += level + self.comment + "Real" + self.endl
    str_out += level + res_name + res_getter + self.real_str + " = "
    str_out += self.zero + self.endl
    for ordi in range(1,self.order+1):
      
      str_out += level +self.comment + "Order " + str(ordi) + self.endl
      dirs = self.name_imdir[ordi]

      for j in range(len(dirs)):
        
        str_out += level + res_name + res_getter + dirs[j] + " = "+ self.zero + self.endl
        
      # end for 

    # end for 

    return str_out
  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def assigno_like_function(self, level = "", f_name = "FUNCTION", lhs_name= "LHS",lhs_ptr=True,
    res_name = "RES", f_open = "(", f_close = ")", to=False, order_specific = False, tab=' '):
    """
    PORPUSE:  Addition like function between two OTIs.
    """
    global h
    str_out = ""

    if to:
      res_getter = self.get_ptr
      res_prev = '*'
    else:
      res_getter = self.get
      res_prev = ''
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if 
    
    leveli = level
    
    if order_specific:
      str_out += leveli + res_prev + res_name + " = "+self.func_name+"_init()"+self.endl
      str_out += leveli + "switch( order ){" + endl
      leveli = level + tab
      str_out += leveli +  "case 0:" + endl
      leveli = level + 2*tab
    # end if

    str_out += leveli + self.comment + "Assign like function \'"
    str_out += f_name + f_open + lhs_name + f_close
    str_out += "\'\n"


    # Write real part.
    str_out += leveli + self.comment + "Real" + self.endl
    str_out += leveli + res_name + res_getter + self.real_str + " = "
    str_out += f_name + f_open
    str_out +=         lhs_name + lhs_getter + self.real_str + f_close + self.endl


    if order_specific:
      str_out += leveli + "break"+self.endl
    # end if 

    for ordi in range(1,self.order+1):
      
      if order_specific:
        leveli = level + tab
        str_out += leveli + "case " + str(ordi) + ":"+endl
        leveli = level + 2*tab
      # end if 

      str_out += leveli +self.comment + "Order " + str(ordi) + self.endl
      dirs = self.name_imdir[ordi]

      for j in range(len(dirs)):
        
        str_out += leveli + res_name + res_getter + dirs[j] + " = "
        str_out += f_name + f_open
        str_out += lhs_name + lhs_getter + dirs[j] 
        str_out += f_close
        str_out += self.endl
        
      # end for 
      
      if order_specific:
        str_out += leveli + "break"+self.endl
      # end if 

    # end for 


    if order_specific:
      leveli = level
      str_out += leveli + "}" + endl
    # end if 

    return str_out
  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def truncim_scalar_function(self, tab = "  ",level = "", lhs_name= "lhs", lhs_ptr=True,
    f_name = "FUNCTION", res_name = "res", f_open = "(", f_close = ")", to=False, deriv= False):
    """
    PORPUSE:  getim like function between two OTIs.
    """
    global h
    str_out = ""

    if to:
      res_getter = self.get_ptr
      res_prev = '*'
    else:
      res_getter = self.get
      res_prev = ''
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
      lhs_prev = '*'
    else:
      lhs_getter = self.get
      lhs_prev = ''
    # end if 

    str_out += level + self.comment + "Copy number." + endl
    str_out += level + res_prev + res_name + " = (" + lhs_prev + lhs_name + ')'+ self.endl

    str_out += level + self.comment + "Truncate all other values when necessary." + endl
    # Write real part.
    str_out += level + self.comment + "Real" + self.endl
    str_out += level + "switch (order){"+endl
    str_out += level + tab +  "case 0:" +endl
    str_out += level + 2*tab + res_prev + res_name + " = " + self.func_name +"_init()" + self.endl
    str_out += level + 2*tab + "break" + self.endl
    

    for ordi in range(1,self.order+1):
      str_out += level + tab + "case "+str(ordi)+":"+endl
      
      dirs = self.name_imdir[ordi]
      levelj = level + 2*tab 
      str_out += levelj + "switch (idx){"+endl 
      for j in range(len(dirs)):
        dir_search  = dirs[j]
        jj = self.idx_imdir[ordi][j]
        str_out += levelj+tab+ "case "+str(jj) + ": " + self.comment + " " + dir_search + endl
        str_out += levelj+2*tab+ res_name + res_getter + dirs[j] + " = " + self.zero + self.endl
        for ordj in range(ordi+1, self.order+1):
          flag = False
          for i in range(len(self.mult_res[ordj])):
            for multiple in self.mult_res[ordj][i]:
              if dir_search in multiple:
                str_out += levelj+2*tab+ res_name + res_getter + self.name_imdir[ordj][i] + " = " + self.zero + self.endl
                break
              # end if 
          # end for 
        # end for 
        str_out += levelj+2*tab+"break"+self.endl
      # end for 
      
      str_out += levelj + "}" + endl
      str_out += levelj + "break"+self.endl
    # end for 

    str_out += level + "}"

    return str_out


  #---------------------------------------------------------------------------------------------------

  #***************************************************************************************************
  def getidx_scalar_function(self, tab = "  ",level = ""):
    """
    PORPUSE:  getim like function between two OTIs.
    """
    global h
    str_out = ""
    
    # Write real part.
    str_out += level + self.comment + "Real" + self.endl
    str_out += level + "switch (order){"+endl
    str_out += level + tab +  "case 0:" +endl
    str_out += level + 2*tab + "return 0" + self.endl
    str_out += level + 2*tab + "break" + self.endl
    
    levelj = level + 2*tab 

    for ordi in range(1,self.order+1):
      str_out += level + tab + "case "+str(ordi)+":"+endl
      
      dirs = self.name_imdir[ordi]
      levelj = level + 2*tab 
      str_out += levelj + "switch (idx){"+endl 
      for j in range(len(dirs)):
        
        jj = self.idx_imdir[ordi][j]
        str_out += levelj+tab+ "case "+str(j) + ":" + endl
        str_out += levelj+2*tab+"return "+str(jj)+self.endl
        str_out += levelj+2*tab+"break"+self.endl
      # end for 
      
      str_out += levelj + "}" + endl
      str_out += levelj + "break"+self.endl
    # end for 

    str_out += levelj + "default:"+endl 
    str_out += levelj+2*tab+"return 0"+self.endl
    str_out += level + "}"

    return str_out
  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def getim_scalar_function(self, tab = "  ",level = "", lhs_name= "lhs", lhs_ptr=True,
    f_name = "FUNCTION", res_name = "res", f_open = "(", f_close = ")", to=False, deriv= False):
    """
    PORPUSE:  getim like function between two OTIs.
    """
    global h
    str_out = ""
    if to:
      res_getter = '*'
    else:
      res_getter = ''
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if 

    if deriv:
      str_out += level + self.coeff_t + " factor"  + self.endl
    # end if 
    str_out += level + self.comment + "Get Imaginary coefficient." + endl
    str_out += level + res_getter + res_name + " = " + self.zero + self.endl

    # Write real part.
    str_out += level + self.comment + "Real" + self.endl
    str_out += level + "switch (order){"+endl
    str_out += level + tab +  "case 0:" +endl
    str_out += level + 2*tab + res_getter + res_name + " = " +lhs_name + lhs_getter + self.real_str + self.endl
    str_out += level + 2*tab + "break" + self.endl
    

    for ordi in range(1,self.order+1):
      str_out += level + tab + "case "+str(ordi)+":"+endl
      
      dirs = self.name_imdir[ordi]
      levelj = level + 2*tab 
      str_out += levelj + "switch (idx){"+endl 
      for j in range(len(dirs)):
        
        jj = self.idx_imdir[ordi][j]
        str_out += levelj+tab+ "case "+str(jj) + ":" + endl
        
        if deriv:
          str_out += levelj+2*tab+"factor = {0:.16e}".format(h.get_deriv_factor(jj,ordi))+self.endl
        # end if 
        
        str_out += levelj+2*tab+res_getter + res_name + " = " + lhs_name + lhs_getter + dirs[j] 

        if deriv:
          str_out += "*factor"
        # end if 

        str_out += self.endl
        str_out += levelj+2*tab+"break"+self.endl
      # end for 
      
      str_out += levelj + "}" + endl
      str_out += levelj + "break"+self.endl
    # end for 

    str_out += level + "}"

    return str_out


  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def setim_scalar_function(self, tab = "  ",level = "", lhs_name= "lhs", lhs_ptr=True,
    f_name = "FUNCTION", res_name = "res", f_open = "(", f_close = ")", to=False):
    """
    PORPUSE:  getim like function between two OTIs.
    """
    global h
    str_out = ""
    
    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if 

    str_out += level + self.comment + "Set Imaginary coefficient." + endl
    # str_out += level + "*" + res_name + " = " + self.zero + self.endl

    # Write real part.
    str_out += level + self.comment + "Real" + self.endl
    str_out += level + "switch (order){"+endl
    str_out += level + tab +  "case 0:" +endl
    str_out += level + 2*tab + lhs_name + lhs_getter + self.real_str + " = " + 'val'  + self.endl
    str_out += level + 2*tab + "break" + self.endl
    

    for ordi in range(1,self.order+1):
      str_out += level + tab + "case "+str(ordi)+":"+endl
      
      dirs = self.name_imdir[ordi]
      levelj = level + 2*tab 
      str_out += levelj + "switch (idx){"+endl 
      for j in range(len(dirs)):
        
        jj = self.idx_imdir[ordi][j]

        str_out += levelj+tab+ "case "+str(jj) + ":" + endl
        str_out += levelj+2*tab+lhs_name + lhs_getter + dirs[j] + " = " +  'val'  + self.endl
        str_out += levelj+2*tab+"break"+self.endl
      # end for 
      
      str_out += levelj + "}" + endl
      str_out += levelj + "break"+self.endl
    # end for 

    str_out += level + "}"

    return str_out


  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def pprint_scalar_function(self, level = "", lhs_name= "lhs", fmt = "%.8g",tab = "  "):
    """
    PORPUSE:  getim like function between two OTIs.
    """

    global h

    str_out = ""
    fmt_im = fmt.replace("%","%+")

    str_out += level + self.comment + "Print function." + endl

    # Write real part.
    str_out += level + self.comment + " Real" + self.endl
    str_out += level + 'printf(\"'+fmt+'\",'
    str_out += lhs_name + self.get_ptr + self.real_str
    str_out += ')'+self.endl
    
    for ordi in range(1,self.order+1):

      str_out += level + self.comment + " Order " + str(ordi)+ self.endl
      dirs = self.name_imdir[ordi]
            
      for j in range(len(dirs)):        
        
        str_out += level + 'printf(\" '+fmt_im+'*' + dirs[j] +'\",'
        str_out += lhs_name + self.get_ptr + dirs[j] 
        str_out += ')'+self.endl
          
      # end for 
    # end for 

    str_out += level + 'printf(\"\\n\")'+self.endl

    return str_out


  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def negation_like_function(self, level = "", f_name = "FUNCTION", lhs_name= "LHS", lhs_ptr=False,
    res_name = "RES", f_open = "(", f_close = ")", to=False):
    """
    PORPUSE:  Negation like function between two OTIs.
    """
    global h
    
    if to:
      res_getter = self.get_ptr
    else:
      res_getter = self.get
    # end if 

    if lhs_ptr:
      lhs_getter = self.get_ptr
    else:
      lhs_getter = self.get
    # end if 


    str_out = ""


    str_out += level + self.comment + "Negation like function \'"
    str_out += f_name + f_open + lhs_name + f_close
    str_out += "\'\n"


    # Write real part.
    str_out += level + self.comment + "Real" + self.endl
    str_out += level + res_name + res_getter + self.real_str + " = "
    str_out += f_name + f_open
    str_out +=         lhs_name + lhs_getter + self.real_str + f_close + self.endl


    for ordi in range(1,self.order+1):
      
      str_out += level +self.comment + "Order " + str(ordi) + self.endl
      dirs = self.name_imdir[ordi]

      for j in range(len(dirs)):
        
        str_out += level + res_name + res_getter + dirs[j] + " = "
        str_out += f_name + f_open
        str_out +=         lhs_name + lhs_getter + dirs[j] + f_close + self.endl
        
      # end for 

    # end for 

    return str_out


  #--------------------------------------------------------------------------------------------------- 


  #***************************************************************************************************
  def write_util_function_getidx(self,level = 0, tab = " "):

    str_out = ""
    leveli = level

    
    lhs = "lhs"
  
    f_prev = self.func_name
    
    lhs_t = self.type_name
    
    func_name = f_prev + "_get_indx" 

    # Write function start.
    str_out += leveli*tab
    leveli += 1
    

    
    func_header  = "imdir_t " + func_name + "(imdir_t idx, ord_t order)"

    self.function_list.append(func_header)
    self.function_list_header['base'].append(func_header)

    str_out += func_header +"{"+endl
    str_out += endl

    str_out += self.getidx_scalar_function( tab = tab,level = leveli*tab)

    str_out += endl


    # Write function end.
    str_out += leveli*tab + 'return 0' + self.endl
    leveli -= 1
    str_out += leveli*tab + '}' + endl

    return str_out
  #---------------------------------------------------------------------------------------------------

  #***************************************************************************************************
  def write_scalar_function_print(self, level = 0, tab = " "):

    str_out = ""
    leveli = level

    
    lhs = "lhs"
  
    f_prev = self.func_name
    
    lhs_t = self.type_name
    
    func_name = f_prev + "_print" 

    # Write function start.
    str_out += leveli*tab
    leveli += 1
    

    
    func_header  = "void " + func_name + "("
    func_header += leveli*tab + lhs_t + "* "+ lhs 
    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header['base'].append(func_header)

    str_out += func_header +"{"+endl
    str_out += endl

    str_out += self.pprint_scalar_function( tab = tab,level = leveli*tab,  lhs_name = lhs)

    str_out += endl


    # Write function end.
    leveli -= 1
    str_out += leveli*tab + '}' + endl

    return str_out
  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def write_scalar_getitem(self, function_name = "get_item", is_elemental = True, level = 0, tab = " ", 
    f_name = "FUNCTION",  separator = ",", lhs_type= "O", lhs_ptr = False, res_type = 'r',
    f_open = "(", f_close = ")", addition = " + ",generator = None, to = False, deriv= False,
    overload = None, write_charact=True ):
    """
    Write Univariate function.

    This module writes the definition of the function, its inputs and output. The generator defines the
    operations within the function block.

    """

    str_out = ""
    leveli = level

    res = "res"
    lhs = "lhs"

    res_tp = self.type_names[res_type]

    f_prev = self.func_name
    
    func_name = f_prev + "_" + function_name 
    
    # Write function start.
    str_out += leveli*tab
    leveli += 1
    
    func_header = ''

    if to:
      func_header += "void "
      func_name += "_to"
    else:
      func_header += res_tp + " "
    # end if 


    func_header +=  func_name + "(" + "imdir_t idx, ord_t order, "
    func_header +=  self.type_name + "* " + lhs

    if to:
      func_header += ", "+res_tp + "* " + res
    # end if 

    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header['base'].append(func_header)

    str_out += func_header +"{"+endl+endl

    if not to:
      str_out += leveli*tab + res_tp + " " + res + self.endl
      str_out += endl
    # end if

    str_out += generator(level = leveli*tab, res_name = res, lhs_name=lhs, lhs_ptr=True, tab= tab,
      deriv=deriv, to=to)
    

    str_out += endl
    # Write function end.

    if not to:
      str_out += leveli*tab + 'return ' + res + self.endl + endl
    # end if 
    leveli -= 1
    str_out += leveli*tab + '}' + endl

    return str_out
  #--------------------------------------------------------------------------------------------------- 


  #***************************************************************************************************
  def write_scalar_taylorint(self, function_name = "taylor_integrate", is_elemental = True, level = 0, tab = " ", 
    f_name = "FUNCTION",  separator = ",", lhs_type= "O", lhs_ptr = False,
    f_open = "(", f_close = ")", addition = " + ",generator = None, to = False,
    overload = None, write_charact=True ):
    """
    Write Univariate function.

    This module writes the definition of the function, its inputs and output. The generator defines the
    operations within the function block.

    """

    str_out = ""
    leveli = level

    res = "res"
    lhs = "lhs"

    f_prev = self.func_name
    
    func_name = f_prev + "_" + function_name 
    
    # Write function start.
    str_out += leveli*tab
    leveli += 1
    
    func_header = ''
    if to:
      func_header += "void "
      func_name   += "_to"
    else:
      func_header += self.type_name + " "
    # end if 

    func_header +=  func_name + "(" + self.coeff_t+"* deltas,"
    func_header +=  self.type_name + "* " + lhs

    if to:
      func_header += ", " + self.type_name + "* " + res
    #end if 
    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header['base'].append(func_header)

    str_out += func_header +"{"+endl+endl

    if not to:
      str_out += leveli*tab + self.type_name + " " + res + self.endl
      str_out += endl
    #end if 

    str_out += generator(level = leveli*tab, res_name = res, lhs_name=lhs, lhs_ptr=True,  
      to=to, f_name = "", f_open = "", f_close = "")
    

    str_out += endl
    # Write function end.

    if not to:
      str_out += leveli*tab + 'return ' + res + self.endl + endl
    # end if 

    leveli -= 1
    str_out += leveli*tab + '}' + endl

    return str_out
  #--------------------------------------------------------------------------------------------------- 





  #***************************************************************************************************
  def write_scalar_get_order_im(self, function_name = "get_order_im", is_elemental = True, level = 0, tab = " ", 
    f_name = "FUNCTION",  separator = ",", lhs_type= "O", lhs_ptr = False, f_open = "(", f_close = ")", 
    addition = " + ",generator = None, to = False, overload = None, write_charact=True ):
    """
    Write Univariate function.

    This module writes the definition of the function, its inputs and output. The generator defines the
    operations within the function block.

    """

    str_out = ""
    leveli = level

    res = "res"
    lhs = "lhs"

    f_prev = self.func_name
    
    func_name = f_prev + "_" + function_name 
    
    # Write function start.
    str_out += leveli*tab
    leveli += 1
    
    func_header = ''

    if to:
      func_header += "void "
      func_name   += "_to"
    else:
      func_header += self.type_name + " " 
    # end if 

    func_header +=  func_name + "(" + "ord_t order, "
    func_header +=  self.type_name + "* " + lhs

    if to:
      func_header += ", "+self.type_name + "* " + res
    # end if 

    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header['base'].append(func_header)

    str_out += func_header +"{"+endl+endl

    if not to:
      str_out += leveli*tab + self.type_name + " " + res + self.endl

    str_out += endl
    
    str_out += generator(level = leveli*tab, res_name = res, lhs_name=lhs, lhs_ptr=True, tab= tab, 
      order_specific = True, f_name=f_name, f_open=f_open, f_close=f_close, to=to )
    

    str_out += endl
    # Write function end.

    if not to:
      str_out += leveli*tab + 'return ' + res + self.endl + endl
    # end if 

    leveli -= 1
    str_out += leveli*tab + '}' + endl

    return str_out
  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def write_scalar_feval(self, function_name = "feval", is_elemental = True, level = 0, tab = " ", 
    f_name = "FUNCTION",  separator = ",", lhs_type= "O", lhs_ptr = False,
    f_open = "(", f_close = ")", addition = " + ", to = False,
    overload = None, write_charact=True ):
    """
    Write Univariate function.

    This module writes the definition of the function, its inputs and output. The generator defines the
    operations within the function block.

    """

    str_out = ""
    leveli = level

    res = "res"
    lhs = "x"

    lhs_t = self.type_names[lhs_type]
    f_post = lhs_type
   
    if lhs_ptr == True:
      lhs_t += '*'
    # end if

    f_prev = self.func_name
    
    func_name = f_prev + "_" + function_name 
    
    # Write function start.
    str_out += leveli*tab
    leveli += 1
    
    func_header = ''

    if to:
      func_header += 'void '
      func_name += '_to'
    else:
      func_header += self.type_name + " "
    # end if 

    func_header +=  func_name + "(" + self.coeff_t + "* feval_re, "
    func_header +=  lhs_t + " " + lhs

    if to:
      func_header += ", "
      func_header += self.type_name+"* "+res
    # end if 

    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header['algebra'].append(func_header)

    str_out += func_header +"{"+endl+endl
    
    if not to:

      str_out += leveli*tab + self.type_name + " " + res + self.endl
      str_out += endl
    
    # end if 

    str_out += self.feval_function_o(level = leveli*tab, res_name = res, lhs_name=lhs, lhs_ptr=True, to=to)
    

    str_out += endl
    # Write function end.

    if to:
      
      leveli -= 1
      str_out += leveli*tab + '}' + endl

    else :

      str_out += leveli*tab + 'return ' + res + self.endl + endl
      leveli -= 1
      str_out += leveli*tab + '}' + endl
    # end if

    return str_out
  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def write_scalar_setitem(self, function_name = "setim", is_elemental = True, level = 0, tab = " ", 
    f_name = "FUNCTION",  separator = ",", lhs_type= "O", lhs_ptr = False,
    f_open = "(", f_close = ")", addition = " + ", overload = None, write_charact=True ):
    """
    Write Univariate function.

    This module writes the definition of the function, its inputs and output. The generator defines the
    operations within the function block.

    """

    str_out = ""
    leveli = level

    res = "res"
    lhs = "x"

    to = False

    lhs_t = self.type_names[lhs_type]
    f_post = lhs_type
   
    if lhs_ptr == True:
      lhs_t += '*'
    # end if

    f_prev = self.func_name
    
    func_name = f_prev + "_" + function_name 
    
    # Write function start.
    str_out += leveli*tab
    leveli += 1
    
    func_header = ''

    func_header += 'void '

    func_header +=  func_name + "(" +self.coeff_t+ " val, imdir_t idx, ord_t order, "
    func_header +=  self.type_name + "* " + lhs

    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header['base'].append(func_header)

    str_out += func_header +"{"+endl+endl
    


    str_out += self.setim_scalar_function(level = leveli*tab, res_name = res, lhs_name=lhs, lhs_ptr=True, to=to)

    str_out += endl
    # Write function end.
      
    leveli -= 1
    str_out += leveli*tab + '}' + endl


    return str_out
  #--------------------------------------------------------------------------------------------------- 



  #***************************************************************************************************
  def write_scalar_init(self, function_name = "init", level = 0, tab = " ", generator = None):
    """
    Write Univariate function.

    This module writes the definition of the function, its inputs and output. The generator defines the
    operations within the function block.

    """

    str_out = ""
    leveli = level

    res = "res"
    f_prev = self.func_name
    
    func_name = f_prev + "_" + function_name 
    
    # Write function start.
    str_out += leveli*tab
    leveli += 1
    
    func_header = ''


    func_header += self.type_name + " "


    func_header +=  func_name + "(void"    
    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header['base'].append(func_header)

    str_out += func_header +"{"+endl+endl

    str_out += leveli*tab + self.type_name + " " + res + self.endl
    str_out += endl

    str_out += generator(level = leveli*tab, res_name = res)

    str_out += endl
    # Write function end.

    str_out += leveli*tab + 'return ' + res + self.endl + endl
    leveli -= 1
    str_out += leveli*tab + '}' + endl

    return str_out
  #--------------------------------------------------------------------------------------------------- 


  #***************************************************************************************************
  def write_get_order_address(self, level = 0, tab = " ", header = 'base'):

    str_out = ""
    leveli = level
    f_post = 'o'

    
    lhs = "num"
    res = "res"
    f_prev = self.func_name
    lhs_t = self.type_names['o']
    func_name = self.func_name + "_get_order_address" 

    # Write function start.
    str_out += leveli*tab
    leveli += 1
    
    func_header = self.coeff_t+'* '    

    func_header +=  func_name + "("
    func_header += "ord_t order, "
    func_header += lhs_t + "* "+ lhs 
    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header[header].append(func_header)

    str_out += func_header +"{"+endl+endl

    str_out += leveli*tab + self.coeff_t + "* " + res + self.endl
    str_out += endl
    
    str_out += leveli*tab + 'switch(order){'+ endl

    for ordi in range(self.order+1):
      str_out += (leveli+1)*tab + 'case '+str(ordi)+':'+ endl
      ndir = len(self.name_imdir[ordi])
      str_out += (leveli+2)*tab + res + ' = &' + lhs + self.get_ptr + self.name_imdir[ordi][0]+ self.endl
      str_out += (leveli+2)*tab + 'break' + self.endl
    # end for
    str_out += leveli*tab + '}'+ endl
  

    str_out += leveli*tab + 'return ' + res + self.endl + endl
    leveli -= 1
    str_out += leveli*tab + '}' + endl


    return str_out
  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def write_get_ndir(self, level = 0, tab = " ", header = 'base', total = False ):

    str_out = ""
    leveli = level
    f_post = 'o'

    
    lhs = "num"
    res = "res"
    f_prev = self.func_name
    lhs_t = self.type_names['o']
    func_name = self.func_name + "_get_ndir_" 

    if total:
      func_name += 'total'
    else:
      func_name += 'order'
    # end if 
    
    # Write function start.
    str_out += leveli*tab
    leveli += 1
    
    func_header = 'ndir_t '    

    func_header +=  func_name + "("
    if not total:
      func_header += "ord_t order, "
    # end if 
    func_header += lhs_t + "* "+ lhs 
    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header[header].append(func_header)

    str_out += func_header +"{"+endl+endl

    str_out += leveli*tab + "ndir_t " + res + self.endl
    str_out += endl
    
    if total:
      ndir = 0
      for ordi in range(self.order):
        ndir += len(self.name_imdir[ordi])
      # end for
      str_out += leveli*tab + res + ' = ' + str(ndir) + self.endl
    else:

      str_out += leveli*tab + 'switch(order){'+ endl

      for ordi in range(self.order+1):
        str_out += (leveli+1)*tab + 'case '+str(ordi)+':'+ endl
        ndir = len(self.name_imdir[ordi])
        str_out += (leveli+2)*tab + res + ' = ' + str(ndir) + self.endl
        str_out += (leveli+2)*tab + 'break' + self.endl
      # end for

      str_out += leveli*tab + '}'+ endl
    # end if 

    str_out += endl
    # Write function end.


    str_out += leveli*tab + 'return ' + res + self.endl + endl
    leveli -= 1
    str_out += leveli*tab + '}' + endl


    return str_out
  #--------------------------------------------------------------------------------------------------- 


  #***************************************************************************************************
  def write_scalar_univar(self, function_name = "NEG", is_elemental = True, level = 0, tab = " ", 
    f_name = "FUNCTION",  separator = ",", lhs_type= "O", lhs_ptr = False,
    f_open = "(", f_close = ")", addition = " + ",generator = None, to = False,
    overload = None, write_charact=True, header = 'base' ):

    """
    Write Univariate function.

    This module writes the definition of the function, its inputs and output. The generator defines the
    operations within the function block.

    """

    str_out = ""
    leveli = level
    f_post = 'o'

    
    lhs = "lhs"
    rhs = "rhs"
    res = "res"
    f_prev = self.func_name
    lhs_t = self.type_names[lhs_type]
    f_post = lhs_type
   
    if lhs_ptr == True:
      lhs_t += '*'
    # end if


    func_name = f_prev + "_" + function_name 
    if write_charact:
      func_name += "_" + f_post
    # end if 


    # Write function start.
    str_out += leveli*tab
    leveli += 1
    
    func_header = ''

    if to:
      func_header += 'void '
      if function_name != 'set':
        func_name += '_to'
      # end if 
    else:
      func_header += self.type_name + " "
    # end if 

    func_header +=  func_name + "("
    func_header += leveli*tab + lhs_t + " "+ lhs 
    
    if to:
      func_header += ", "
      func_header += self.type_name+"* "+res
    # end if 
    
    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header[header].append(func_header)

    str_out += func_header +"{"+endl+endl

    if not to:

      str_out += leveli*tab + self.type_name + " " + res + self.endl
      str_out += endl
    
    # end if 

    str_out += generator(f_name = f_name, level = leveli*tab, f_open = f_open, 
      f_close =f_close, res_name = res, lhs_name = lhs, lhs_ptr=lhs_ptr, to=to)


    str_out += endl
    # Write function end.

    
    if to:
      
      leveli -= 1
      str_out += leveli*tab + '}' + endl

    else :

      str_out += leveli*tab + 'return ' + res + self.endl + endl
      leveli -= 1
      str_out += leveli*tab + '}' + endl
    # end if

    return str_out
  #--------------------------------------------------------------------------------------------------- 


  #***************************************************************************************************
  def write_scalar_function(self, function_name = "FUNCTION", is_elemental = True, level = 0, tab = " ", 
    f_name = "FUNCTION", lhs_type= "O", lhs_ptr=False, rhs_type= "O", rhs_ptr=False, separator = ",", 
    f_open = "(", f_close = ")", addition = " + ",generator = None, to=False,
    overload = None, write_charact=True, header = 'base' ):

    str_out = ""
    leveli = level

    lhs = "lhs"
    rhs = "rhs"
    res = "res"

    f_prev = self.func_name
    
    lhs_t  = self.type_names[lhs_type]
    f_post = lhs_type
  
    if lhs_ptr == True:
      lhs_t += '*'
    # end if

    rhs_t  = self.type_names[rhs_type]
    f_post += rhs_type
  
    if rhs_ptr == True:
      rhs_t += '*'
    # end if

    func_name = f_prev + "_" + function_name 
    
    if write_charact:
      func_name += "_"+ f_post
    # end if 

    # Write function start.
    str_out += leveli*tab
    leveli += 1
    
    func_header = ''

    if to:
      func_header += 'void '
      func_name += '_to'
    else:
      func_header += self.type_name + " "
    # end if 

    func_header += func_name + "("
    func_header += leveli*tab + lhs_t + " "+ lhs + ','
    func_header += leveli*tab + rhs_t + " "+ rhs 

    if to:
      func_header += ", "
      func_header += self.type_name+"* "+res
    # end if 

    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header[header].append(func_header)
    
    str_out += func_header +"{"+endl

    if not to:
      str_out += leveli*tab + self.type_name + " " + res + self.endl
      str_out += endl
    # end if 

    str_out += generator(f_name = f_name, separator = separator,  
               level = leveli*tab, f_open = f_open, f_close =f_close, res_name = res,
               lhs_name = lhs, lhs_ptr=lhs_ptr, rhs_name=rhs, rhs_ptr=rhs_ptr, to=to)


    str_out += endl
    # Write function end.

    if not to:  
      str_out += leveli*tab + 'return ' + res + self.endl + endl
    # end if

    leveli -= 1
    str_out += leveli*tab + '}' + endl

    return str_out
  #--------------------------------------------------------------------------------------------------- 





  #***************************************************************************************************
  def write_truncscalar_function(self, function_name = "FUNCTION", is_elemental = True, level = 0, tab = " ", 
    f_name = "FUNCTION", lhs_type= "O", lhs_ptr=False, rhs_type= "O", rhs_ptr=False, separator = ",", 
    f_open = "(", f_close = ")", addition = " + ",generator = None, to=False,
    overload = None, write_charact=True, header = 'base' ):

    str_out = ""
    leveli = level

    lhs = "lhs"
    rhs = "rhs"
    res = "res"

    f_prev = self.func_name
    
    lhs_t  = self.type_names[lhs_type]
    f_post = lhs_type
  
    if lhs_ptr == True:
      lhs_t += '*'
    # end if

    rhs_t  = self.type_names[rhs_type]
    f_post += rhs_type
  
    if rhs_ptr == True:
      rhs_t += '*'
    # end if

    func_name = f_prev + "_" + function_name 
    
    if write_charact:
      func_name += "_"+ f_post
    # end if 

    # Write function start.
    str_out += leveli*tab
    leveli += 1
    
    func_header = ''

    if to:
      func_header += 'void '
      func_name += '_to'
    else:
      func_header += self.type_name + " "
    # end if 

    func_header += func_name + "(" + "ord_t order, "
    func_header += lhs_t + " "+ lhs + ','
    func_header += rhs_t + " "+ rhs 

    if to:
      func_header += ", "
      func_header += self.type_name+"* "+res
    # end if 

    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header[header].append(func_header)
    
    str_out += func_header +"{"+endl

    if not to:
      str_out += leveli*tab + self.type_name + " " + res + self.endl
      str_out += endl
    # end if 

    str_out += generator(f_name = f_name, separator = separator,  
               level = leveli*tab, f_open = f_open, f_close =f_close, res_name = res, tab=tab,
               lhs_name = lhs, lhs_ptr=lhs_ptr, rhs_name=rhs, rhs_ptr=rhs_ptr, to=to)


    str_out += endl
    # Write function end.

    if not to:  
      str_out += leveli*tab + 'return ' + res + self.endl + endl
    # end if

    leveli -= 1
    str_out += leveli*tab + '}' + endl

    return str_out
  #--------------------------------------------------------------------------------------------------- 









  #***************************************************************************************************
  def write_scalar_trivar(self, function_name = "FUNCTION", is_elemental = True, level = 0, tab = " ", 
    f_name = "FUNCTION", a_type= "o", a_ptr=False, b_type= "o", b_ptr=False, c_type= "o", c_ptr=False,
    separator = ",", f_open = "(", f_close = ")", addition = " + ",generator = None, to=False,
    overload = None, write_charact=True, header = 'base' ):

    str_out = ""
    leveli = level

    a = "a"
    b = "b"
    c = "c"
    res = "res"

    f_prev = self.func_name
    
    a_t  = self.type_names[a_type]
    f_post = a_type
  
    if a_ptr == True:
      a_t += '*'
    # end if

    b_t  = self.type_names[b_type]
    f_post += b_type
  
    if b_ptr == True:
      b_t += '*'
    # end if

    c_t  = self.type_names[c_type]
    # f_post += c_type
  
    if c_ptr == True:
      c_t += '*'
    # end if

    func_name = f_prev + "_" + function_name 
    
    if write_charact:
      func_name += "_"+ f_post
    # end if 

    # Write function start.
    str_out += leveli*tab
    leveli += 1
    
    func_header = ''

    if to:
      func_header += 'void '
      func_name += '_to'
    else:
      func_header += self.type_name + " "
    # end if 

    func_header += func_name + "("
    func_header += leveli*tab + a_t + " "+ a + ','
    func_header += leveli*tab + b_t + " "+ b + ','
    func_header += leveli*tab + c_t + " "+ c 

    if to:
      func_header += ", "
      func_header += self.type_name+"* "+res
    # end if 

    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header[header].append(func_header)
    
    str_out += func_header +"{"+endl

    if not to:
      str_out += leveli*tab + self.type_name + " " + res + self.endl
      str_out += endl
    # end if 

    str_out += generator(f_name = f_name, separator = separator,  
               level = leveli*tab, f_open = f_open, f_close =f_close, res_name = res,
               a_name = a, a_ptr=a_ptr, b_name=b, b_ptr=b_ptr, c_name=c, c_ptr=c_ptr, to=to)


    str_out += endl
    # Write function end.

    if not to:  
      str_out += leveli*tab + 'return ' + res + self.endl + endl
    # end if

    leveli -= 1
    str_out += leveli*tab + '}' + endl

    return str_out
  #--------------------------------------------------------------------------------------------------- 



  #***************************************************************************************************
  def write_scalar_truncgem(self, function_name = "FUNCTION", is_elemental = True, level = 0, tab = " ", 
    f_name = "FUNCTION", a_type= "o", a_ptr=False, b_type= "o", b_ptr=False, c_type= "o", c_ptr=False,
    separator = ",", f_open = "(", f_close = ")", addition = " + ",generator = None, to=False,
    overload = None, write_charact=True, header = 'base' ):

    str_out = ""
    leveli = level

    a = "a"
    b = "b"
    c = "c"
    res = "res"

    f_prev = self.func_name
    
    a_t  = self.type_names[a_type]
    f_post = a_type
  
    if a_ptr == True:
      a_t += '*'
    # end if

    b_t  = self.type_names[b_type]
    f_post += b_type
  
    if b_ptr == True:
      b_t += '*'
    # end if

    c_t  = self.type_names[c_type]
    # f_post += c_type
  
    if c_ptr == True:
      c_t += '*'
    # end if

    func_name = f_prev + "_" + function_name 
    
    if write_charact:
      func_name += "_"+ f_post
    # end if 

    # Write function start.
    str_out += leveli*tab
    leveli += 1
    
    func_header = ''

    if to:
      func_header += 'void '
      func_name += '_to'
    else:
      func_header += self.type_name + " "
    # end if 

    func_header += func_name + "( ord_t ord_lhs,"
    func_header += leveli*tab + a_t + " "+ a + ','
    func_header += leveli*tab + 'ord_t ord_rhs,'
    func_header += leveli*tab + b_t + " "+ b + ','
    func_header += leveli*tab + c_t + " "+ c 

    if to:
      func_header += ", "
      func_header += self.type_name+"* "+res
    # end if 

    func_header += ")"

    self.function_list.append(func_header)
    self.function_list_header[header].append(func_header)
    
    str_out += func_header +"{"+endl

    if not to:
      str_out += leveli*tab + self.type_name + " " + res + self.endl
      str_out += endl
    # end if 

    str_out += generator(f_name = f_name, separator = separator,  tab=tab,
               level = leveli*tab, f_open = f_open, f_close =f_close, res_name = res,
               a_name = a, a_ptr=a_ptr, b_name=b, b_ptr=b_ptr, c_name=c, c_ptr=c_ptr, to=to)


    str_out += endl
    # Write function end.

    if not to:  
      str_out += leveli*tab + 'return ' + res + self.endl + endl
    # end if

    leveli -= 1
    str_out += leveli*tab + '}' + endl

    return str_out
  #--------------------------------------------------------------------------------------------------- 








  #***************************************************************************************************
  def write_files(self, modulename = None, tab = '  ', base_dir=''):
    """
    PORPUSE:  Write file of module containing OTI operations.
    """    
    str_out = ""

    if modulename is not None:
      mname = modulename
    else:
      mname = self.func_name.lower()
    # end if 


    src_path = getpath()+"source_conv/"
    files=[]
    
    # Supported extensions.
    supp_ext = ['h','pxd','pxi','c','pyx']

    # r=>root, d=>directories, f=>files
    for r, d, f in os.walk(src_path):
      for item in f:
        ext = item.split('.')[-1]
        if ext in supp_ext:
          files.append(os.path.join(r.replace(src_path,""), item))
        # end if 
      # end for 
    # end for
    

    # 1. Write module name if in fortran ...
    level = 0

    # Define type
    num_struct = self.set_type_c( level = level*tab ) + endl
    
    base_str = self.gen_base_file( level = level, tab = tab)
    base_include = self.write_header( header = 'base')
    
    algebra_str = self.gen_algebra_file( level = level, tab = tab)

    algebra_include = self.write_header( header = 'algebra')
    arr_get_all_derivs_str = self.gen_get_all_derivs_array(level = "  ", tab = "  ")
    for file in files:
      # Write the header file

      src_file = src_path + file

      print("Processing: ",file)

      f = open(src_file, "r" )
      f_str=f.read()
      f.close()

      exp_file = base_dir+'/'+file.replace("number",mname)
      print(exp_file)
      os.makedirs(os.path.dirname(exp_file), exist_ok=True)
      
      f_exp = f_str.format(  num_type   = self.type_name,       num_func   = self.func_name,       
        num_pytype   = self.pytype_name,       num_pyfunc   = self.pyfunc_name, 
        fenum_type = self.type_name_fe,    fenum_func = self.func_name_fe,    
        fenum_pytype = self.pytype_name_fe,    fenum_pyfunc = self.pyfunc_name_fe,    
        arr_type   = self.type_name_arr,   arr_func   = self.func_name_arr,   
        arr_pytype   = self.pytype_name_arr,   arr_pyfunc   = self.pyfunc_name_arr,   
        fearr_type = self.type_name_fearr, fearr_func = self.func_name_fearr, 
        fearr_pytype = self.pytype_name_fearr, fearr_pyfunc = self.pyfunc_name_fearr, 
        oti_order = self.order, real_str = self.real_str, module_name = mname, 
        module_name_upper = mname.upper(), base_src = base_str, algebra_src = algebra_str,
        base_include = base_include, algebra_include = algebra_include, 
        utils_include = "",  num_struct = num_struct, nbases = self.nbases, arr_get_all_derivs = arr_get_all_derivs_str)

      f = open(exp_file, "w" )
      f.write(f_exp)
      f.close()

    # end for 

    self.process_headers_static( base_dir = base_dir, tab=tab)
    self.process_static_include_files( base_dir = base_dir)

  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def write_header(self, header = ''):
    
    header_file = ''
    for funct in self.function_list_header[header]:
      header_file += funct+";"+endl
    # end for 

    return header_file
  #--------------------------------------------------------------------------------------------------- 

  
  #***************************************************************************************************
  def check_file_and_dirs(self, filename):
    dir_name = os.path.dirname(filename)
    if dir_name != '':
      if not os.path.exists(dir_name):
        try:
          os.makedirs(dir_name)
        except OSError as exc: # Guard against race condition
          if exc.errno != errno.EEXIST:
              raise

  #--------------------------------------------------------------------------------------------------- 


  #***************************************************************************************************
  def gen_base_file(self, level = 0, tab = ' '):
    

    contents = ""
    
    # Init function
    contents+=self.write_scalar_init(function_name = "init", level = level, tab = tab, 
      generator = self.init_like_function)
    contents += endl

    
    # Get ndirs
    contents+=self.write_get_ndir(level = level, tab = tab, header = 'base', total = False )
    contents += endl

    contents+=self.write_get_ndir(level = level, tab = tab, header = 'base', total = True )
    contents += endl

    # get order address
    contents+=self.write_get_order_address(level = level, tab = tab, header = 'base')
    contents += endl
    

    # Standard assignment
    contents += self.write_scalar_univar(function_name = "create", is_elemental = True, level = level, 
      tab = tab, lhs_type = 'r', f_name = "",   f_open = "", header = 'base',
      f_close = "", overload = "=",generator = self.assignr_like_function)
    contents += endl

    contents += self.write_scalar_univar(function_name = "set", is_elemental = True, level = level, 
      tab = tab, lhs_type = 'r', f_name = "",   f_open = "", to=True, header = 'base',
      f_close = "", overload = "=",generator = self.assignr_like_function)
    contents += endl

    contents += self.write_scalar_univar(function_name = "set", is_elemental = True, level = level, 
      tab = tab, lhs_type = 'o', lhs_ptr = True, f_name = "",   f_open = "", to=True, header = 'base',
      f_close = "", overload = "=",generator = self.assigno_like_function)
    contents += endl

    contents += self.write_scalar_get_order_im(function_name = "get_order_im", is_elemental = True, level = level, 
      tab = tab, f_name = "",  separator = "", lhs_type= "o", lhs_ptr = True, f_open = "", f_close = "", 
    addition = " + ",generator = self.assigno_like_function, to = False )
    contents += endl

    contents += self.write_scalar_get_order_im(function_name = "get_order_im", is_elemental = True, level = level, 
      tab = tab, f_name = "",  separator = "", lhs_type= "o", lhs_ptr = True, f_open = "", f_close = "", 
    addition = " + ",generator = self.assigno_like_function, to = True )
    contents += endl

    # Get item
    contents += self.write_scalar_getitem(function_name = "get_item",level = level, tab = tab, 
      lhs_type = 'o', lhs_ptr=True, generator = self.getim_scalar_function)
    contents += endl 


    # Get Deriv
    contents += self.write_scalar_getitem(function_name = "get_deriv",level = level, tab = tab, deriv=True,
      lhs_type = 'o', lhs_ptr=True, generator = self.getim_scalar_function)
    contents += endl 


    # Set item
    contents += self.write_scalar_setitem(function_name = "set_item",level = level, tab = tab, 
      lhs_type = 'o', lhs_ptr=True)
    contents += endl 

    # Copy functions
    contents += self.write_scalar_univar(function_name = "copy", is_elemental = True, level = level, 
      tab = tab, lhs_type = 'o',lhs_ptr=True, f_name = "",   f_open = "",  to=False,header = 'base',
      f_close = "", overload = "=",generator = self.assigno_like_function, write_charact=False)
    contents += endl
    contents += self.write_scalar_univar(function_name = "copy", is_elemental = True, level = level, 
      tab = tab, lhs_type = 'o',lhs_ptr=True, f_name = "",   f_open = "", to=True,header = 'base',
      f_close = "", overload = "=",generator = self.assigno_like_function, write_charact=False)
    contents += endl

    # Taylor integration
    contents += self.write_scalar_taylorint(function_name = "taylor_integrate",level = level, tab = tab, 
      lhs_type = 'o', lhs_ptr=True, generator = self.taylorint_like_function, to=False)
    contents += endl 

    contents += self.write_scalar_taylorint(function_name = "taylor_integrate",level = level, tab = tab, 
      lhs_type = 'o', lhs_ptr=True, generator = self.taylorint_like_function, to=True)
    contents += endl 

    # Truncate
    contents += self.write_scalar_getitem(function_name = "truncate_im",level = level, tab = tab, 
      lhs_type = 'o', lhs_ptr=True, generator = self.truncim_scalar_function, res_type = 'o')
    contents += endl 

    contents += self.write_scalar_getitem(function_name = "truncate_im",level = level, tab = tab, 
      lhs_type = 'o', lhs_ptr=True, generator = self.truncim_scalar_function, to = True, res_type = 'o')
    contents += endl 


    # Print scalar
    contents += self.write_scalar_function_print( level = level, tab = tab)
    contents += endl   


    # Get index
    contents += self.write_util_function_getidx( level = level, tab = tab)
    contents += endl  

    return contents

  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def gen_algebra_file(self, level = 0, tab = ' '):
    

    append_header = 'algebra'

    contents = ""

    # Standard NEGATION
    contents += self.write_scalar_univar(function_name = "neg", is_elemental = True, level = level, 
      tab = tab, f_name = "",   f_open = "-", lhs_type='o', lhs_ptr=True, to=False, write_charact=False,
      f_close = "", overload = "-",generator = self.negation_like_function,header = 'algebra')
    contents += endl

    contents += self.write_scalar_univar(function_name = "neg", is_elemental = True, level = level, 
      tab = tab, f_name = "",   f_open = "-", lhs_type='o', lhs_ptr=True, to=True, write_charact=False,
      f_close = "", overload = "-",generator = self.negation_like_function,header = 'algebra')
    contents += endl


    # Standard ADDITION
    contents += self.write_scalar_function(function_name = "sum", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "o", rhs_type= "o", separator = " + ", f_open = "", 
      lhs_ptr=True,rhs_ptr=True,header = 'algebra',
      f_close = "", generator = self.addition_like_function_oo, overload = "+")
    contents += endl

    contents += self.write_scalar_function(function_name = "sum", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "o", rhs_type= "o", separator = " + ", f_open = "", 
      lhs_ptr=True,rhs_ptr=True,header = 'algebra',
      f_close = "", generator = self.addition_like_function_oo, overload = "+", to=True)
    contents += endl

    contents += self.write_scalar_function(function_name = "sum", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= self.real_str, rhs_type= "o", separator = " + ", f_open = "", 
      lhs_ptr=False, rhs_ptr=True,header = 'algebra',
      f_close = "", generator = self.addition_like_function_ro, overload = "+" )
    contents += endl

    contents += self.write_scalar_function(function_name = "sum", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= self.real_str, rhs_type= "o", separator = " + ", f_open = "", 
      lhs_ptr=False, rhs_ptr=True,header = 'algebra',
      f_close = "", generator = self.addition_like_function_ro, overload = "+" , to=True)
    contents += endl


    # Standard SUBTRACTION
    contents += self.write_scalar_function(function_name = "sub", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "o", rhs_type= "o", separator = " - ", f_open = "", 
      lhs_ptr=True, rhs_ptr= True, to=False,header = 'algebra',
      f_close = "", generator = self.addition_like_function_oo, overload = "-" )
    contents += endl

    contents += self.write_scalar_function(function_name = "sub", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "o", rhs_type= "o", separator = " - ", f_open = "", 
      lhs_ptr=True, rhs_ptr= True, to=True,header = 'algebra',
      f_close = "", generator = self.addition_like_function_oo, overload = "-" )
    contents += endl

    contents += self.write_scalar_function(function_name = "sub", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type=  "r", rhs_type= "o", separator = " - ", 
      lhs_ptr=False, rhs_ptr= True, to=False,header = 'algebra',
      f_open = "", f_close = "", generator = self.addition_like_function_ro, overload = "-" )
    contents += endl

    contents += self.write_scalar_function(function_name = "sub", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type=  "r", rhs_type= "o", separator = " - ", 
      lhs_ptr=False, rhs_ptr= True, to=True,header = 'algebra',
      f_open = "", f_close = "", generator = self.addition_like_function_ro, overload = "-" )
    contents += endl

    contents += self.write_scalar_function(function_name = "sub", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "o", rhs_type=  "r", separator = " - ", f_open = "", 
      lhs_ptr=True, rhs_ptr= False, to=False,header = 'algebra',
      f_close = "", generator = self.addition_like_function_or, overload = "-")
    contents += endl

    contents += self.write_scalar_function(function_name = "sub", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "o", rhs_type=  "r", separator = " - ", f_open = "", 
      lhs_ptr=True, rhs_ptr= False, to=True,header = 'algebra',
      f_close = "", generator = self.addition_like_function_or, overload = "-")
    contents += endl












    # Standard MULTIPLICATION
    contents += self.write_scalar_function(function_name = "mul", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "o", rhs_type= "o", separator = " * ", f_open = "", 
      lhs_ptr=True, rhs_ptr= True, to=False,header = 'algebra',
      f_close = "", generator = self.multiplication_like_function_oo, overload = "*" )
    contents += endl

    contents += self.write_scalar_function(function_name = "mul", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "o", rhs_type= "o", separator = " * ", f_open = "", 
      lhs_ptr=True, rhs_ptr= True, to=True,header = 'algebra',
      f_close = "", generator = self.multiplication_like_function_oo, overload = "*" )
    contents += endl

    contents += self.write_scalar_function(function_name = "mul", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "r", rhs_type= "o", separator = " * ", f_open = "", 
      lhs_ptr=False, rhs_ptr= True, to=False,header = 'algebra',
      f_close = "", generator = self.multiplication_like_function_ro, overload = "*" )
    contents += endl

    contents += self.write_scalar_function(function_name = "mul", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "r", rhs_type= "o", separator = " * ", f_open = "", 
      lhs_ptr=False, rhs_ptr= True, to=True,header = 'algebra',
      f_close = "", generator = self.multiplication_like_function_ro, overload = "*" )
    contents += endl

      

    # Truncated MULTIPLICATION
    contents += self.write_scalar_function(function_name = "trunc_mul", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "o", rhs_type= "o", separator = " * ", f_open = "", 
      lhs_ptr=True, rhs_ptr= True, to=False,header = 'algebra',
      f_close = "", generator = self.truncmul_like_function_oo, overload = "*" )
    contents += endl

    contents += self.write_scalar_function(function_name = "trunc_mul", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "o", rhs_type= "o", separator = " * ", f_open = "", 
      lhs_ptr=True, rhs_ptr= True, to=True,header = 'algebra',
      f_close = "", generator = self.truncmul_like_function_oo, overload = "*" )
    contents += endl





    # General multiplication a * b + c
    contents += self.write_scalar_trivar(function_name = "gem", is_elemental = True, level = level, 
      tab = tab, f_name = "", a_type= "o", b_type= "o", c_type = "o", 
      a_ptr = True, b_ptr = True, c_ptr = True, to=False, 
      separator = " * ", f_open = "",header = 'algebra', 
      f_close = "", generator = self.gem_like_function_oo, overload = "*", write_charact=True )
    contents += endl

    contents += self.write_scalar_trivar(function_name = "gem", is_elemental = True, level = level, 
      tab = tab, f_name = "", a_type= "o", b_type= "o", c_type = "o", 
      a_ptr = True, b_ptr = True, c_ptr = True, to=True, 
      separator = " * ", f_open = "",header = 'algebra', 
      f_close = "", generator = self.gem_like_function_oo, overload = "*", write_charact=True )
    contents += endl

    contents += self.write_scalar_trivar(function_name = "gem", is_elemental = True, level = level, 
      tab = tab, f_name = "", a_type= "r", b_type= "o", c_type = "o", 
      a_ptr = False, b_ptr = True, c_ptr = True, to=False, 
      separator = " * ", f_open = "",header = 'algebra', 
      f_close = "", generator = self.gem_like_function_ro, overload = "*", write_charact=True )
    contents += endl

    contents += self.write_scalar_trivar(function_name = "gem", is_elemental = True, level = level, 
      tab = tab, f_name = "", a_type= "r", b_type= "o", c_type = "o", 
      a_ptr = False, b_ptr = True, c_ptr = True, to=True, 
      separator = " * ", f_open = "",header = 'algebra', 
      f_close = "", generator = self.gem_like_function_ro, overload = "*", write_charact=True )
    contents += endl

    # Truncated GEM
    contents += self.write_scalar_truncgem(function_name = "trunc_gem", is_elemental = True, level = level, 
      tab = tab, f_name = "", a_type= "o", b_type= "o", c_type = "o", 
      a_ptr = True, b_ptr = True, c_ptr = True, to=True, 
      separator = " * ", f_open = "",header = 'algebra', 
      f_close = "", generator = self.truncgem_like_function_oo, overload = "*", write_charact=True )
    contents += endl

    # Truncated addition
    contents += self.write_truncscalar_function(function_name = "trunc_sum", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "o", rhs_type= "o", separator = " + ", f_open = "", 
      lhs_ptr=True, rhs_ptr= True, to=True,header = 'algebra',
      f_close = "", generator = self.truncaddition_like_function_oo, overload = "+" )
    contents += endl

    # Truncated negation
    contents += self.write_truncscalar_function(function_name = "trunc_sub", is_elemental = True, level = level, 
      tab = tab, f_name = "", lhs_type= "o", rhs_type= "o", separator = " - ", f_open = "", 
      lhs_ptr=True, rhs_ptr= True, to=True,header = 'algebra',
      f_close = "", generator = self.truncaddition_like_function_oo, overload = "-" )
    contents += endl
  
    

    #Function EVALuation
    contents += self.write_scalar_feval(function_name = "feval", is_elemental = True, level = level, tab = tab, 
      lhs_type= "o", lhs_ptr = True, to = False )
    contents += endl

    contents += self.write_scalar_feval(function_name = "feval", is_elemental = True, level = level, tab = tab, 
      lhs_type= "o", lhs_ptr = True, to = True )
    contents += endl

    return contents
  #---------------------------------------------------------------------------------------------------
  #***************************************************************************************************
  def _get_sparse_sq_info(self, basis_d):
    """
    Compute sparse multiplication info for squaring diffs[basis_d].

    diffs[d] has only 2 non-zero components: r and e_{basis_d}.
    diffs[d]² has only 3 non-zero: r, e_{basis_d}, e_{basis_d basis_d}.

    Returns:
      nz_a: set of field names non-zero in diffs[d]
      sq_nz: set of field names non-zero in diffs[d]²
      sq_terms: dict {field_name: [(lhs_field, rhs_field), ...]} for diffs[d]²
      mul_terms: dict {field_name: [(lhs_field, rhs_field), ...]} for diffs[d]² * dense_b
    """
    # Non-zero components of diffs[d]: real + one 1st-order component
    nz_a = set()
    nz_a.add('r')
    if self.order >= 1 and basis_d < len(self.name_imdir[1]):
      nz_a.add(self.name_imdir[1][basis_d])

    # Compute which components of a² are non-zero
    sq_nz = set()
    sq_terms = {}
    for ord_out in range(self.order + 1):
      for comp_idx in range(len(self.name_imdir[ord_out])):
        comp_name = self.name_imdir[ord_out][comp_idx]
        mults = self.mult_res_total[ord_out][comp_idx]
        surviving = []
        for term in mults:
          lhs_name = term[0][2]
          rhs_name = term[1][2]
          if lhs_name in nz_a and rhs_name in nz_a:
            surviving.append((lhs_name, rhs_name))
        if surviving:
          sq_nz.add(comp_name)
          sq_terms[comp_name] = surviving

    # For (a² * b) where b is dense: keep terms where lhs is in sq_nz
    mul_terms = {}
    for ord_out in range(self.order + 1):
      for comp_idx in range(len(self.name_imdir[ord_out])):
        comp_name = self.name_imdir[ord_out][comp_idx]
        mults = self.mult_res_total[ord_out][comp_idx]
        surviving = []
        for term in mults:
          lhs_name = term[0][2]
          rhs_name = term[1][2]
          if lhs_name in sq_nz:
            surviving.append((lhs_name, rhs_name))
        if surviving:
          mul_terms[comp_name] = surviving

    return nz_a, sq_nz, sq_terms, mul_terms

  #---------------------------------------------------------------------------------------------------
  #***************************************************************************************************
  def gen_get_all_derivs_array(self, level="  ", tab="  "):
    """
    PURPOSE: Generate the get_all_derivs function for array class (omat/omatm*n*).
             This generates Cython code that extracts all derivatives into a 3D numpy array.
    """
    global endl
    
    str_out = ""
    
    str_out += level + "#***************************************************************************************************" + endl
    str_out += level + "def get_all_derivs(self, bases_t nbasis, ord_t order):" + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "This method creates a new 3D numpy array with all derivatives exported from" + endl
    str_out += level + tab + "the current OTI matso object. A new dimension is added to account for each" + endl
    str_out += level + tab + "derivative. Derivatives are sorted associating the same index-order as" + endl
    str_out += level + tab + "the OTI number structure." + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "The total number of derivatives is given by the (nbasis,order) pair" + endl
    str_out += level + tab + "to be exported." + endl
    str_out += level + tab + ":param nbasis: Number of bases in the OTI number (Related with the number" + endl
    str_out += level + tab + "               of variables)." + endl
    str_out += level + tab + ":param order: Maximum order of directions to export (max order of" + endl
    str_out += level + tab + "              derivative)." + endl
    str_out += level + tab + ":returns: Returns a new numpy array with the derivative information. The" + endl
    str_out += level + tab + "          resulting array contains the derivatives as follows:" + endl
    str_out += level + tab + "          A = X.get_all_derivs(nbasis, order)" + endl
    str_out += level + tab + "          The shape of A is [nDirTotal, X.shape[0], X.shape[1]], where" + endl
    str_out += level + tab + "          nDirTotal is the total number of imaginary directions of nbasis" + endl
    str_out += level + tab + "          and order." + endl
    str_out += level + tab + "          A[:,i,j] contains all the imaginary coefficients for the" + endl
    str_out += level + tab + "          component X[i,j]." + endl
    str_out += level + tab + "          -> A[0,i,j]: contains the real part of X[i,j]." + endl
    str_out += level + tab + "          -> A[1,i,j]: contains the epsilon_1 part of X[i,j]. And so on." + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "#***************************************************************************" + endl
    str_out += level + tab + "global dhl" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "cdef double[:,:,:] res" + endl
    str_out += level + tab + "cdef coeff_t factor" + endl
    str_out += level + tab + "cdef ndir_t ndir, ndir_ord" + endl
    str_out += level + tab + "cdef ord_t ordi" + endl
    str_out += level + tab + "cdef " + self.type_name + " num" + endl
    str_out += level + tab + "cdef imdir_t idx" + endl
    str_out += level + tab + "cdef int64_t i, j, kk, count" + endl
    str_out += level + tab + "cdef coeff_t coeff" + endl
    str_out += level + tab + "cdef uint64_t ncols = self.arr.ncols" + endl
    str_out += level + tab + "cdef uint64_t size = self.arr.size" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "# Get first the number of imaginary directions total" + endl
    str_out += level + tab + "ndir = dhelp_ndirTotal(nbasis, order)" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "res = np.zeros((ndir, self.arr.nrows, self.arr.ncols), dtype=np.float64)" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "# Release the GIL for the main loop" + endl
    str_out += level + tab + "with nogil:" + endl
    str_out += level + tab + tab + "" + endl
    str_out += level + tab + tab + "for kk in range(size):" + endl
    str_out += level + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + "i = kk // ncols" + endl
    str_out += level + tab + tab + tab + "j = kk % ncols" + endl
    str_out += level + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + "# Get the OTI number at position (i,j)" + endl
    str_out += level + tab + tab + tab + "num = " + self.func_name_arr + "_get_item_ij(&self.arr, i, j)" + endl
    str_out += level + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + "# Set real part (order 0)" + endl
    str_out += level + tab + tab + tab + "res[0, i, j] = num.r" + endl
    str_out += level + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + "# Set derivative coefficients for each order" + endl
    str_out += level + tab + tab + tab + "count = 1" + endl
    str_out += level + tab + tab + tab + "for ordi in range(1, order + 1):" + endl
    str_out += level + tab + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + tab + "ndir_ord = dhelp_ndirOrder(nbasis, ordi)" + endl
    str_out += level + tab + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + tab + "for idx in range(ndir_ord):" + endl
    str_out += level + tab + tab + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + tab + tab + "# Get the raw imaginary coefficient" + endl
    str_out += level + tab + tab + tab + tab + tab + "coeff = " + self.func_name + "_get_item(idx, ordi, &num)" + endl
    str_out += level + tab + tab + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + tab + tab + "# Get the derivative factor (factorial scaling)" + endl
    str_out += level + tab + tab + tab + tab + tab + "factor = dhelp_get_deriv_factor(idx, ordi, dhl)" + endl
    str_out += level + tab + tab + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + tab + tab + "# Store scaled derivative" + endl
    str_out += level + tab + tab + tab + tab + tab + "res[count + idx, i, j] = factor * coeff" + endl
    str_out += level + tab + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + tab + "# end for" + endl
    str_out += level + tab + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + tab + "count = count + ndir_ord" + endl
    str_out += level + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + "# end for" + endl
    str_out += level + tab + tab + "" + endl
    str_out += level + tab + tab + "# end for" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "# end with" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "return np.asarray(res)" + endl
    str_out += level + "" + endl
    str_out += level + "#---------------------------------------------------------------------------------------------------" + endl

    # ── get_all_derivs_into: write into pre-allocated buffer ─────────────────
    str_out += level + "#***************************************************************************************************" + endl
    str_out += level + "def get_all_derivs_into(self, bases_t nbasis, ord_t order, double[:,:,:] out):" + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "Same as get_all_derivs but writes into a pre-allocated output array." + endl
    str_out += level + tab + "The 'out' array must have shape (nDirTotal, nrows, ncols) and dtype float64." + endl
    str_out += level + tab + "The array is zeroed and filled in-place. Returns the same array." + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "global dhl" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "cdef coeff_t factor" + endl
    str_out += level + tab + "cdef ndir_t ndir, ndir_ord" + endl
    str_out += level + tab + "cdef ord_t ordi" + endl
    str_out += level + tab + "cdef " + self.type_name + " num" + endl
    str_out += level + tab + "cdef imdir_t idx" + endl
    str_out += level + tab + "cdef int64_t i, j, kk, count" + endl
    str_out += level + tab + "cdef coeff_t coeff" + endl
    str_out += level + tab + "cdef uint64_t ncols = self.arr.ncols" + endl
    str_out += level + tab + "cdef uint64_t size = self.arr.size" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "ndir = dhelp_ndirTotal(nbasis, order)" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "# Zero the output buffer" + endl
    str_out += level + tab + "cdef uint64_t ii" + endl
    str_out += level + tab + "for ii in range(ndir * self.arr.nrows * self.arr.ncols):" + endl
    str_out += level + tab + tab + "(&out[0,0,0])[ii] = 0.0" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "with nogil:" + endl
    str_out += level + tab + tab + "" + endl
    str_out += level + tab + tab + "for kk in range(size):" + endl
    str_out += level + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + "i = kk // ncols" + endl
    str_out += level + tab + tab + tab + "j = kk % ncols" + endl
    str_out += level + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + "num = " + self.func_name_arr + "_get_item_ij(&self.arr, i, j)" + endl
    str_out += level + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + "out[0, i, j] = num.r" + endl
    str_out += level + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + "count = 1" + endl
    str_out += level + tab + tab + tab + "for ordi in range(1, order + 1):" + endl
    str_out += level + tab + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + tab + "ndir_ord = dhelp_ndirOrder(nbasis, ordi)" + endl
    str_out += level + tab + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + tab + "for idx in range(ndir_ord):" + endl
    str_out += level + tab + tab + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + tab + tab + "coeff = " + self.func_name + "_get_item(idx, ordi, &num)" + endl
    str_out += level + tab + tab + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + tab + tab + "factor = dhelp_get_deriv_factor(idx, ordi, dhl)" + endl
    str_out += level + tab + tab + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + tab + tab + "out[count + idx, i, j] = factor * coeff" + endl
    str_out += level + tab + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + tab + "count = count + ndir_ord" + endl
    str_out += level + tab + tab + tab + "" + endl
    str_out += level + tab + tab + "" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "return np.asarray(out)" + endl
    str_out += level + "" + endl
    str_out += level + "#---------------------------------------------------------------------------------------------------" + endl

    # ── get_all_derivs_fast: direct memory cast, no get_item switch ─────────
    str_out += level + "#***************************************************************************************************" + endl
    str_out += level + "def get_all_derivs_fast(self, double[:] factors, double[:,:,:] out):" + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "Fast derivative extraction via direct struct memory access." + endl
    str_out += level + tab + "Casts each onumm_t struct to double* and reads coefficients directly," + endl
    str_out += level + tab + "avoiding the per-coefficient get_item switch/lookup overhead." + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "Args:" + endl
    str_out += level + tab + "  factors: 1D array of length ndir with precomputed derivative factors." + endl
    str_out += level + tab + "  out: 3D array of shape (ndir, nrows, ncols) to write results into." + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "cdef uint64_t ncols = self.arr.ncols" + endl
    str_out += level + tab + "cdef uint64_t nrows = self.arr.nrows" + endl
    str_out += level + tab + "cdef uint64_t size = self.arr.size" + endl
    str_out += level + tab + "cdef uint64_t ndir = factors.shape[0]" + endl
    str_out += level + tab + "cdef " + self.type_name + "* p_data = self.arr.p_data" + endl
    str_out += level + tab + "cdef double* coeffs" + endl
    str_out += level + tab + "cdef double* fptr = &factors[0]" + endl
    str_out += level + tab + "cdef double* optr = &out[0, 0, 0]" + endl
    str_out += level + tab + "cdef uint64_t kk, d, i, j, plane_size" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "plane_size = nrows * ncols" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "with nogil:" + endl
    str_out += level + tab + tab + "for kk in range(size):" + endl
    str_out += level + tab + tab + tab + "# Cast struct to contiguous double array" + endl
    str_out += level + tab + tab + tab + "coeffs = <double*>&p_data[kk]" + endl
    str_out += level + tab + tab + tab + "" + endl
    str_out += level + tab + tab + tab + "for d in range(ndir):" + endl
    str_out += level + tab + tab + tab + tab + "optr[d * plane_size + kk] = fptr[d] * coeffs[d]" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "return np.asarray(out)" + endl
    str_out += level + "" + endl
    str_out += level + "#---------------------------------------------------------------------------------------------------" + endl
    str_out += level + "" + endl

    # ── vdot_expand_fast: fused vdot(W, expand(self)) without materializing expand ────────
    str_out += level + "def vdot_expand_fast(self, double[:] factors, double[:,:,:] W):" + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "Fused vdot(W, expand(self)) without materializing the expanded array." + endl
    str_out += level + tab + "Computes sum_d sum_ij W[d,i,j] * factors[d] * coeffs[d] directly." + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "Args:" + endl
    str_out += level + tab + "  factors: 1D array of length ndir with precomputed derivative factors." + endl
    str_out += level + tab + "  W: 3D array of shape (ndir, nrows, ncols) to dot against." + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "Returns:" + endl
    str_out += level + tab + "  Scalar double result of the dot product." + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "cdef uint64_t size = self.arr.size" + endl
    str_out += level + tab + "cdef uint64_t nrows = self.arr.nrows" + endl
    str_out += level + tab + "cdef uint64_t ncols = self.arr.ncols" + endl
    str_out += level + tab + "cdef uint64_t ndir = factors.shape[0]" + endl
    str_out += level + tab + "cdef " + self.type_name + "* p_data = self.arr.p_data" + endl
    str_out += level + tab + "cdef double* coeffs" + endl
    str_out += level + tab + "cdef double* fptr = &factors[0]" + endl
    str_out += level + tab + "cdef double* wptr = &W[0, 0, 0]" + endl
    str_out += level + tab + "cdef uint64_t kk, d, plane_size" + endl
    str_out += level + tab + "cdef double result = 0.0" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "plane_size = nrows * ncols" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "with nogil:" + endl
    str_out += level + tab + tab + "for kk in range(size):" + endl
    str_out += level + tab + tab + tab + "coeffs = <double*>&p_data[kk]" + endl
    str_out += level + tab + tab + tab + "for d in range(ndir):" + endl
    str_out += level + tab + tab + tab + tab + "result += wptr[d * plane_size + kk] * fptr[d] * coeffs[d]" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "return result" + endl
    str_out += level + "" + endl
    str_out += level + "#---------------------------------------------------------------------------------------------------" + endl

    # ── fused_sqdist: compute sqdist = Σ ell_sq[d] * diff[d]² in one C-level pass ─────────
    str_out += level + "" + endl
    str_out += level + "#***************************************************************************************************" + endl
    str_out += level + "def fused_sqdist(self, list diffs, double[:] ell_sq):" + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "Fused squared-distance computation for SE/RQ kernels." + endl
    str_out += level + tab + "Computes: self[kk] = Σ_d ell_sq[d] * diff[d][kk]²" + endl
    str_out += level + tab + "where the squaring and accumulation use OTI arithmetic." + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "Eliminates 3*dim Python→C transitions per kernel evaluation" + endl
    str_out += level + tab + "and avoids intermediate array allocations." + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "Args:" + endl
    str_out += level + tab + "  diffs: list of " + self.pytype_name_arr + " arrays (differences_by_dim)." + endl
    str_out += level + tab + "  ell_sq: 1D memoryview of ell[d]² values (length == len(diffs))." + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "cdef uint64_t size = self.arr.size" + endl
    str_out += level + tab + "cdef uint64_t dim = <uint64_t>len(diffs)" + endl
    str_out += level + tab + "cdef uint64_t kk, d" + endl
    str_out += level + tab + "cdef " + self.type_name + " acc, tmp" + endl
    str_out += level + tab + "cdef " + self.type_name + "* res_data = self.arr.p_data" + endl
    str_out += level + tab + "cdef double* ell_ptr = &ell_sq[0]" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "# Build C array of pointers to diff data" + endl
    str_out += level + tab + "cdef " + self.type_name + "** diff_ptrs = <" + self.type_name + "**>malloc(dim * sizeof(" + self.type_name + "*))" + endl
    str_out += level + tab + "if diff_ptrs == NULL:" + endl
    str_out += level + tab + tab + 'raise MemoryError("Failed to allocate diff pointer array")' + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "cdef " + self.pytype_name_arr + " diff_arr" + endl
    str_out += level + tab + "for d in range(dim):" + endl
    str_out += level + tab + tab + "diff_arr = <" + self.pytype_name_arr + ">diffs[d]" + endl
    str_out += level + tab + tab + "diff_ptrs[d] = diff_arr.arr.p_data" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "try:" + endl
    str_out += level + tab + tab + "with nogil:" + endl
    str_out += level + tab + tab + tab + "for kk in range(size):" + endl
    str_out += level + tab + tab + tab + tab + "acc = " + self.func_name + "_init()" + endl
    str_out += level + tab + tab + tab + tab + "for d in range(dim):" + endl
    str_out += level + tab + tab + tab + tab + tab + self.func_name + "_mul_oo_to(&diff_ptrs[d][kk], &diff_ptrs[d][kk], &tmp)" + endl
    str_out += level + tab + tab + tab + tab + tab + self.func_name + "_mul_ro_to(ell_ptr[d], &tmp, &tmp)" + endl
    str_out += level + tab + tab + tab + tab + tab + self.func_name + "_sum_oo_to(&acc, &tmp, &acc)" + endl
    str_out += level + tab + tab + tab + tab + "res_data[kk] = acc" + endl
    str_out += level + tab + "finally:" + endl
    str_out += level + tab + tab + "free(diff_ptrs)" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "return self" + endl
    str_out += level + "" + endl
    str_out += level + "#---------------------------------------------------------------------------------------------------" + endl

    # ── fused_scale_sq_mul: compute self = scale * a² * b in one C-level pass ─────────
    str_out += level + "" + endl
    str_out += level + "#***************************************************************************************************" + endl
    str_out += level + "def fused_scale_sq_mul(self, " + self.pytype_name_arr + " a, " + self.pytype_name_arr + " b, double scale):" + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "Fused computation: self[kk] = scale * a[kk]² * b[kk]." + endl
    str_out += level + tab + "Used for SE/RQ gradient terms: dphi_d = -ln10 * ell_d² * diff_d² * phi." + endl
    str_out += level + tab + "Replaces 3 separate mul calls and 2 intermediate arrays." + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "cdef uint64_t size = self.arr.size" + endl
    str_out += level + tab + "cdef uint64_t kk" + endl
    str_out += level + tab + "cdef " + self.type_name + " tmp, tmp2" + endl
    str_out += level + tab + "cdef " + self.type_name + "* res_data = self.arr.p_data" + endl
    str_out += level + tab + "cdef " + self.type_name + "* a_data = a.arr.p_data" + endl
    str_out += level + tab + "cdef " + self.type_name + "* b_data = b.arr.p_data" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "with nogil:" + endl
    str_out += level + tab + tab + "for kk in range(size):" + endl
    str_out += level + tab + tab + tab + self.func_name + "_mul_oo_to(&a_data[kk], &a_data[kk], &tmp)  # tmp = a²" + endl
    str_out += level + tab + tab + tab + self.func_name + "_mul_oo_to(&tmp, &b_data[kk], &tmp2)  # tmp2 = a² * b" + endl
    str_out += level + tab + tab + tab + self.func_name + "_mul_ro_to(scale, &tmp2, &res_data[kk])  # res = scale * a² * b" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "return self" + endl
    str_out += level + "" + endl
    str_out += level + "#---------------------------------------------------------------------------------------------------" + endl

    # ── fused_sum_sq: compute self = Σ_d diff[d]² in one C-level pass (no ell scaling) ────
    str_out += level + "" + endl
    str_out += level + "#***************************************************************************************************" + endl
    str_out += level + "def fused_sum_sq(self, list diffs):" + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "Fused sum-of-squares: self[kk] = Σ_d diff[d][kk]²." + endl
    str_out += level + tab + "Used for isotropic kernel gradients where all length scales are equal." + endl
    str_out += level + tab + "No ell scaling — just OTI squaring and accumulation." + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "cdef uint64_t size = self.arr.size" + endl
    str_out += level + tab + "cdef uint64_t dim = <uint64_t>len(diffs)" + endl
    str_out += level + tab + "cdef uint64_t kk, d" + endl
    str_out += level + tab + "cdef " + self.type_name + " acc, tmp" + endl
    str_out += level + tab + "cdef " + self.type_name + "* res_data = self.arr.p_data" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "# Build C array of pointers to diff data" + endl
    str_out += level + tab + "cdef " + self.type_name + "** diff_ptrs = <" + self.type_name + "**>malloc(dim * sizeof(" + self.type_name + "*))" + endl
    str_out += level + tab + "if diff_ptrs == NULL:" + endl
    str_out += level + tab + tab + 'raise MemoryError("Failed to allocate diff pointer array")' + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "cdef " + self.pytype_name_arr + " diff_arr" + endl
    str_out += level + tab + "for d in range(dim):" + endl
    str_out += level + tab + tab + "diff_arr = <" + self.pytype_name_arr + ">diffs[d]" + endl
    str_out += level + tab + tab + "diff_ptrs[d] = diff_arr.arr.p_data" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "try:" + endl
    str_out += level + tab + tab + "with nogil:" + endl
    str_out += level + tab + tab + tab + "for kk in range(size):" + endl
    str_out += level + tab + tab + tab + tab + "acc = " + self.func_name + "_init()" + endl
    str_out += level + tab + tab + tab + tab + "for d in range(dim):" + endl
    str_out += level + tab + tab + tab + tab + tab + self.func_name + "_mul_oo_to(&diff_ptrs[d][kk], &diff_ptrs[d][kk], &tmp)" + endl
    str_out += level + tab + tab + tab + tab + tab + self.func_name + "_sum_oo_to(&acc, &tmp, &acc)" + endl
    str_out += level + tab + tab + tab + tab + "res_data[kk] = acc" + endl
    str_out += level + tab + "finally:" + endl
    str_out += level + tab + tab + "free(diff_ptrs)" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "return self" + endl
    str_out += level + "" + endl
    str_out += level + "#---------------------------------------------------------------------------------------------------" + endl

    # ── SPARSE fused functions: exploit diffs[d] sparsity ─────────────────────────
    # diffs[d] has only 2 non-zero components (r and e_d), so diffs[d]² has only 3.
    # These sparse variants inline the minimal arithmetic per basis, avoiding the
    # generic mul_oo_to which blindly computes all components.

    # Collect all component field names in struct order
    all_comps = []
    for ordi in range(self.order + 1):
      all_comps.extend(self.name_imdir[ordi])

    # ── fused_sqdist_sparse ──────────────────────────────────────────────────────
    # Loop-swapped: d outside kk to eliminate branching from the hot inner loop.
    str_out += level + "" + endl
    str_out += level + "#***************************************************************************************************" + endl
    str_out += level + "def fused_sqdist_sparse(self, list diffs, double[:] ell_sq):" + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "Sparse fused squared-distance: self[kk] = sum_d ell_sq[d] * diff[d][kk]^2." + endl
    str_out += level + tab + "Exploits that diffs[d] has only 2 non-zero OTI components (r, e_d)," + endl
    str_out += level + tab + "so the square has only 3 non-zero components instead of " + str(len(all_comps)) + "." + endl
    str_out += level + tab + "Loop order: d outside, kk inside — no branching in the hot loop." + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "cdef uint64_t size = self.arr.size" + endl
    str_out += level + tab + "cdef uint64_t dim = <uint64_t>len(diffs)" + endl
    str_out += level + tab + "cdef uint64_t kk" + endl
    str_out += level + tab + "cdef " + self.type_name + "* res_data = self.arr.p_data" + endl
    str_out += level + tab + "cdef double* ell_ptr = &ell_sq[0]" + endl
    str_out += level + tab + "cdef double ell_d, a_r, a_ed" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "# Build C array of pointers to diff data" + endl
    str_out += level + tab + "cdef " + self.type_name + "** diff_ptrs = <" + self.type_name + "**>malloc(dim * sizeof(" + self.type_name + "*))" + endl
    str_out += level + tab + "if diff_ptrs == NULL:" + endl
    str_out += level + tab + tab + 'raise MemoryError("Failed to allocate diff pointer array")' + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "cdef " + self.pytype_name_arr + " diff_arr" + endl
    str_out += level + tab + "for d in range(dim):" + endl
    str_out += level + tab + tab + "diff_arr = <" + self.pytype_name_arr + ">diffs[d]" + endl
    str_out += level + tab + tab + "diff_ptrs[d] = diff_arr.arr.p_data" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "try:" + endl
    str_out += level + tab + tab + "with nogil:" + endl
    # Zero pass — one sweep over the data
    str_out += level + tab + tab + tab + "for kk in range(size):" + endl
    for comp in all_comps:
      str_out += level + tab + tab + tab + tab + "res_data[kk]." + comp + " = 0.0" + endl
    str_out += level + tab + tab + tab + "" + endl
    # Per-basis accumulation — each basis gets its own tight kk loop
    for basis_d in range(self.nbases):
      nz_a, sq_nz, sq_terms, _ = self._get_sparse_sq_info(basis_d)
      ed_name = self.name_imdir[1][basis_d]
      str_out += level + tab + tab + tab + "if dim > " + str(basis_d) + ":" + endl
      indent = level + tab + tab + tab + tab
      str_out += indent + "ell_d = ell_ptr[" + str(basis_d) + "]" + endl
      str_out += indent + "for kk in range(size):" + endl
      indent2 = indent + tab
      str_out += indent2 + "a_r = diff_ptrs[" + str(basis_d) + "][kk].r" + endl
      str_out += indent2 + "a_ed = diff_ptrs[" + str(basis_d) + "][kk]." + ed_name + endl
      for comp in all_comps:
        if comp in sq_terms:
          terms = sq_terms[comp]
          expr_parts = []
          for (lhs, rhs) in terms:
            lhs_var = "a_r" if lhs == "r" else "a_ed"
            rhs_var = "a_r" if rhs == "r" else "a_ed"
            expr_parts.append(lhs_var + " * " + rhs_var)
          expr = " + ".join(expr_parts)
          str_out += indent2 + "res_data[kk]." + comp + " += ell_d * (" + expr + ")" + endl
    str_out += level + tab + "finally:" + endl
    str_out += level + tab + tab + "free(diff_ptrs)" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "return self" + endl
    str_out += level + "" + endl
    str_out += level + "#---------------------------------------------------------------------------------------------------" + endl

    # ── fused_sum_sq_sparse ──────────────────────────────────────────────────────
    # Loop-swapped: d outside kk to eliminate branching from the hot inner loop.
    str_out += level + "" + endl
    str_out += level + "#***************************************************************************************************" + endl
    str_out += level + "def fused_sum_sq_sparse(self, list diffs):" + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "Sparse fused sum-of-squares: self[kk] = sum_d diff[d][kk]^2." + endl
    str_out += level + tab + "Same sparsity exploitation as fused_sqdist_sparse but without ell scaling." + endl
    str_out += level + tab + "Loop order: d outside, kk inside — no branching in the hot loop." + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "cdef uint64_t size = self.arr.size" + endl
    str_out += level + tab + "cdef uint64_t dim = <uint64_t>len(diffs)" + endl
    str_out += level + tab + "cdef uint64_t kk" + endl
    str_out += level + tab + "cdef " + self.type_name + "* res_data = self.arr.p_data" + endl
    str_out += level + tab + "cdef double a_r, a_ed" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "# Build C array of pointers to diff data" + endl
    str_out += level + tab + "cdef " + self.type_name + "** diff_ptrs = <" + self.type_name + "**>malloc(dim * sizeof(" + self.type_name + "*))" + endl
    str_out += level + tab + "if diff_ptrs == NULL:" + endl
    str_out += level + tab + tab + 'raise MemoryError("Failed to allocate diff pointer array")' + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "cdef " + self.pytype_name_arr + " diff_arr" + endl
    str_out += level + tab + "for d in range(dim):" + endl
    str_out += level + tab + tab + "diff_arr = <" + self.pytype_name_arr + ">diffs[d]" + endl
    str_out += level + tab + tab + "diff_ptrs[d] = diff_arr.arr.p_data" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "try:" + endl
    str_out += level + tab + tab + "with nogil:" + endl
    # Zero pass
    str_out += level + tab + tab + tab + "for kk in range(size):" + endl
    for comp in all_comps:
      str_out += level + tab + tab + tab + tab + "res_data[kk]." + comp + " = 0.0" + endl
    str_out += level + tab + tab + tab + "" + endl
    # Per-basis accumulation
    for basis_d in range(self.nbases):
      nz_a, sq_nz, sq_terms, _ = self._get_sparse_sq_info(basis_d)
      ed_name = self.name_imdir[1][basis_d]
      str_out += level + tab + tab + tab + "if dim > " + str(basis_d) + ":" + endl
      indent = level + tab + tab + tab + tab
      str_out += indent + "for kk in range(size):" + endl
      indent2 = indent + tab
      str_out += indent2 + "a_r = diff_ptrs[" + str(basis_d) + "][kk].r" + endl
      str_out += indent2 + "a_ed = diff_ptrs[" + str(basis_d) + "][kk]." + ed_name + endl
      for comp in all_comps:
        if comp in sq_terms:
          terms = sq_terms[comp]
          expr_parts = []
          for (lhs, rhs) in terms:
            lhs_var = "a_r" if lhs == "r" else "a_ed"
            rhs_var = "a_r" if rhs == "r" else "a_ed"
            expr_parts.append(lhs_var + " * " + rhs_var)
          expr = " + ".join(expr_parts)
          str_out += indent2 + "res_data[kk]." + comp + " += " + expr + endl
    str_out += level + tab + "finally:" + endl
    str_out += level + tab + tab + "free(diff_ptrs)" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "return self" + endl
    str_out += level + "" + endl
    str_out += level + "#---------------------------------------------------------------------------------------------------" + endl

    # ── fused_scale_sq_mul_sparse ────────────────────────────────────────────────
    str_out += level + "" + endl
    str_out += level + "#***************************************************************************************************" + endl
    str_out += level + "def fused_scale_sq_mul_sparse(self, " + self.pytype_name_arr + " a, " + self.pytype_name_arr + " b, double scale, uint64_t basis_d):" + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "Sparse fused computation: self[kk] = scale * a[kk]^2 * b[kk]." + endl
    str_out += level + tab + "basis_d identifies which basis direction a=diffs[d] corresponds to," + endl
    str_out += level + tab + "enabling sparse squaring (3 non-zero) and filtered multiplication." + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "cdef uint64_t size = self.arr.size" + endl
    str_out += level + tab + "cdef uint64_t kk" + endl
    str_out += level + tab + "cdef " + self.type_name + "* res_data = self.arr.p_data" + endl
    str_out += level + tab + "cdef " + self.type_name + "* a_data = a.arr.p_data" + endl
    str_out += level + tab + "cdef " + self.type_name + "* b_data = b.arr.p_data" + endl
    str_out += level + tab + "cdef double a_r, a_ed, sq_r, sq_ed, sq_edd" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "with nogil:" + endl
    # Generate one branch per basis — the switch is OUTSIDE the kk loop
    for basis_d in range(self.nbases):
      nz_a, sq_nz, sq_terms, mul_terms = self._get_sparse_sq_info(basis_d)
      ed_name = self.name_imdir[1][basis_d]
      # Find the 2nd-order component name (e_dd)
      edd_name = None
      for comp in sq_nz:
        if comp != 'r' and comp != ed_name:
          edd_name = comp
          break

      if basis_d == 0:
        str_out += level + tab + tab + "if basis_d == 0:" + endl
      else:
        str_out += level + tab + tab + "elif basis_d == " + str(basis_d) + ":" + endl

      indent = level + tab + tab + tab
      str_out += indent + "for kk in range(size):" + endl
      indent2 = indent + tab
      # Extract non-zero components of a into locals
      str_out += indent2 + "a_r = a_data[kk].r" + endl
      str_out += indent2 + "a_ed = a_data[kk]." + ed_name + endl
      # Compute sparse square into locals
      # sq_r = a_r * a_r  (from sq_terms['r'])
      str_out += indent2 + "sq_r = a_r * a_r" + endl
      # sq_ed = terms for the 1st-order component
      if ed_name in sq_terms:
        terms = sq_terms[ed_name]
        expr_parts = []
        for (lhs, rhs) in terms:
          lhs_var = "a_r" if lhs == "r" else "a_ed"
          rhs_var = "a_r" if rhs == "r" else "a_ed"
          expr_parts.append(lhs_var + " * " + rhs_var)
        str_out += indent2 + "sq_ed = " + " + ".join(expr_parts) + endl
      # sq_edd = terms for the 2nd-order component
      if edd_name and edd_name in sq_terms:
        terms = sq_terms[edd_name]
        expr_parts = []
        for (lhs, rhs) in terms:
          lhs_var = "a_r" if lhs == "r" else "a_ed"
          rhs_var = "a_r" if rhs == "r" else "a_ed"
          expr_parts.append(lhs_var + " * " + rhs_var)
        str_out += indent2 + "sq_edd = " + " + ".join(expr_parts) + endl
      str_out += indent2 + "" + endl
      # Now compute res = scale * sq * b using filtered mul_terms
      # Map sq component names to local variable names
      sq_var_map = {'r': 'sq_r'}
      if ed_name:
        sq_var_map[ed_name] = 'sq_ed'
      if edd_name:
        sq_var_map[edd_name] = 'sq_edd'

      for comp in all_comps:
        if comp in mul_terms:
          terms = mul_terms[comp]
          expr_parts = []
          for (lhs, rhs) in terms:
            lhs_var = sq_var_map[lhs]
            rhs_var = "b_data[kk]." + rhs
            expr_parts.append(lhs_var + " * " + rhs_var)
          expr = " + ".join(expr_parts)
          str_out += indent2 + "res_data[kk]." + comp + " = scale * (" + expr + ")" + endl
        else:
          # This component gets zero contribution
          str_out += indent2 + "res_data[kk]." + comp + " = 0.0" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "return self" + endl
    str_out += level + "" + endl
    str_out += level + "#---------------------------------------------------------------------------------------------------" + endl

    # ── fused_from_real_with_perturbations ──────────────────────────────────────
    # Creates an (n1, n2) OTI array where:
    #   out[i,j].r         = real_diffs[i, j]
    #   out[i,j].<im_part> = perturb1[i].<im_part> - perturb2[j].<im_part>
    # This replaces the Python double-loop in differences_by_dim_func across
    # DEGP, DDEGP, and GDDEGP, eliminating oti.zeros + per-row OTI assignment.
    str_out += level + "" + endl
    str_out += level + "#***************************************************************************************************" + endl
    str_out += level + "def fused_from_real_with_perturbations(self, double[:, :] real_diffs, " + self.pytype_name_arr + " perturb1, " + self.pytype_name_arr + " perturb2):" + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "Fused construction of OTI difference array." + endl
    str_out += level + tab + "Sets self[i,j].real = real_diffs[i,j] and" + endl
    str_out += level + tab + "self[i,j].imag = perturb1[i] - perturb2[j] for all imaginary directions." + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "Args:" + endl
    str_out += level + tab + "  real_diffs: 2D memoryview of shape (n1, n2) with real differences." + endl
    str_out += level + tab + "  perturb1: " + self.pytype_name_arr + " of shape (n1, 1) — OTI perturbations for X1." + endl
    str_out += level + tab + "  perturb2: " + self.pytype_name_arr + " of shape (n2, 1) — OTI perturbations for X2." + endl
    str_out += level + tab + '"""' + endl
    str_out += level + tab + "cdef uint64_t n1 = real_diffs.shape[0]" + endl
    str_out += level + tab + "cdef uint64_t n2 = real_diffs.shape[1]" + endl
    str_out += level + tab + "cdef uint64_t i, j, kk" + endl
    str_out += level + tab + "cdef " + self.type_name + "* res_data = self.arr.p_data" + endl
    str_out += level + tab + "cdef " + self.type_name + "* p1_data = perturb1.arr.p_data" + endl
    str_out += level + tab + "cdef " + self.type_name + "* p2_data = perturb2.arr.p_data" + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "with nogil:" + endl
    str_out += level + tab + tab + "for i in range(n1):" + endl
    str_out += level + tab + tab + tab + "for j in range(n2):" + endl
    str_out += level + tab + tab + tab + tab + "kk = i * n2 + j" + endl
    # Set real part from numpy
    str_out += level + tab + tab + tab + tab + "res_data[kk].r = real_diffs[i, j]" + endl
    # Set each imaginary direction: out = perturb1[i] - perturb2[j]
    for ordi in range(1, self.order + 1):
      dirs = self.name_imdir[ordi]
      for d_name in dirs:
        str_out += level + tab + tab + tab + tab + "res_data[kk]." + d_name + " = p1_data[i]." + d_name + " - p2_data[j]." + d_name + endl
    str_out += level + tab + "" + endl
    str_out += level + tab + "return self" + endl
    str_out += level + "" + endl
    str_out += level + "#---------------------------------------------------------------------------------------------------" + endl

    return str_out

  #---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
  #***************************************************************************************************
  def process_static_include_files(self, base_dir = ''):
    
    folders = ['include/oti/',
    'include/pyoti/c_otilib/',
    'include/pyoti/',
    'src/c/',
    ]
    
    cython_types = ['pxi','pxd','pyx']
    c_types = ['h','c']

    c_files = []
    cy_files = []
    # r=>root, d=>directories, f=>files
    for folder in folders:
      src_path = os.path.join(base_dir,folder)
      # print(src_path)
      for r, d, f in os.walk(src_path):        
        for item in f:
          ext = item.split('.')[-1]
          if 'static.' in item:
            if ext in cython_types:
              src_file = os.path.join(r, item)
              walk_folder = os.path.join(r, 'static/')
              cy_files.append([src_file,walk_folder,src_path])
            elif ext in c_types:
              src_file = os.path.join(r, item)
              walk_folder = os.path.join(r, 'static/')
              c_files.append([src_file,walk_folder,src_path])
            # end if 
          # end if 
        # end for 
        break
      # end for
    # end for

    # print(c_files)
    # print()
    # print(cy_files)

    # Replace Cython files:
    for src_file,walk_folder,src_path in cy_files:
      # print(src_file)
      # print(walk)
      str_out = "\n"
      for r, d, f in os.walk(walk_folder): 
        for item in f:
          ext = item.split('.')[-1]
          if ext in cython_types and "__init__" not in item:
            str_out += 'include "{0}"\n\n'.format(os.path.join(r.replace(src_path,""),item))
          # end if
        # end for
        break 
      # end for 
      file = open(src_file,'w')
      file.write(str_out)
      file.close()
    # end for 

    # Replace C files:
    for src_file,walk_folder,src_path in c_files:
      # print(src_file)
      # print(walk)
      name = "OTI_"+src_file.replace(src_path,"").upper().replace('/',"_").replace(".",'_')
      str_out = ''
      if '.h' in src_file:
        str_out += "#ifndef "+name + endl
        str_out += "#define "+name + endl + endl
      # end if 
      for r, d, f in os.walk(walk_folder): 
        for item in f:
          ext = item.split('.')[-1]
          if ext in c_types:
            str_out += '#include "{0}"\n\n'.format(os.path.join(r.replace(src_path,""),item))
          # end if
        # end for
        break 
      # end for 
      if '.h' in src_file:
        str_out += "#endif"+endl
      # end if 
      file = open(src_file,'w')
      file.write(str_out)
      file.close()
    # end for 

    
    
  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def process_headers_static(self, base_dir = '', tab='  '):

    folders={}
    # folders['static'] = []

    
    include_path_val = os.path.join(base_dir,'include/oti/static/')
    for r, d, f in os.walk(include_path_val):
      for direct in d:
        folders[os.path.join(r,direct).replace(include_path_val,"")]=[]
      break
    #  

    # r=>root, d=>directories, f=>files
    for key, files_in_dir in folders.items():
      location = include_path_val + key
      for r, d, f in os.walk(location):
        for item in f:
          if '.h' in item:
            files_in_dir.append(os.path.join(r, item))
          #
        #
      #
    #

    out_dir = os.path.join(base_dir,"include/pyoti/c_otilib/static/")
    InterHeader = """# ========================================================================================="""
         
    for key, files in folders.items():
      structures = {}
      structures['num']   = ""
      structures['fenum'] = ""
      structures['arr']   = ""
      structures['fearr'] = ""

      header_out = ""
      str_out = "" 
      # print(key)
      for file in files:
        # print(files)
        f = open(file,"r")
        string = f.read()
        f.close()
        if "enums" in file:
          header_out += "\n# From "+file+"\n"
          string =  remove_annotations(string).replace("enum", "cdef enum")
          string = string.replace("typedef","ctypedef")
          string = string.replace("}","# }")
          string = string.replace("{",": # {")
          header_out += string
          header_out += "\n"
        elif "structures" in file:
          struct = "\n# From "+file+"\n"
          string =  remove_annotations(string).replace("enum", "cdef enum")
          string = string.replace("typedef","ctypedef")
          pattern = "}(.*?);"
          type_name = re.search(pattern, string).group(1)
          string = string.replace("struct", 'struct '+type_name)
          if 'array' in file:
            if 'gauss' in file:
              val_key = 'fearr'
            else:
              val_key = 'arr'
            # end if
          elif 'scalar' in file:
            if 'gauss' in file:
              val_key = 'fenum'
            else:
              val_key = 'num'
            # end if 
          # end if
          string = string.replace("}","# }")
          string = string.replace("{",": # {")

          structures[val_key] = "\n# From "+file+"\n"
          structures[val_key] += string + "\n"          
        else:
          str_out += "\n# From "+file+"\n"
          string = remove_annotations(string).replace("(void)","()").replace("( void )","()")
          for i in range(10):
            string = string.replace(" \n","\n")
          #
          # string = string.replace(",\n",", &\n")
          str_out += string
          str_out += "\n"
        # end if 
      #
      # Collect the structure files in correct order.
      header_out += structures['num']
      header_out += structures['arr']
      header_out += structures['fenum']
      header_out += structures['fearr']

      str_out_file = header_out +InterHeader+ str_out
      str_out_file = str_out_file.replace("\n","\n"+tab)
      str_out_file = 'cdef extern from "oti/oti.h" nogil:\n\n' + str_out_file
      
      f=open(out_dir+key+".pxi","w")

      f.write(str_out_file)
      f.close()
    # end for 
  #--------------------------------------------------------------------------------------------------- 

  #***************************************************************************************************
  def clean_output_folder_files(self, base_dir = ''):
    
    import shutil

    folders = ['include/oti/static/',
    'include/pyoti/c_otilib/static/',
    'include/pyoti/static/',
    'src/c/static/',
    'src/python/pyoti/cython/static/',
    ]
    tab = '  '
    # r=>root, d=>directories, f=>files
    for folder in folders:
      src_path = os.path.join(base_dir,folder)
      # print(src_path)
      for r, d, f in os.walk(src_path):        
        for item in f:
          ext = item.split('.')[-1]
          if '__init__' not in item:
            os.remove( os.path.join(r,item) )
          # end if 
        # end for 
        for item in d:
          ext = item.split('.')[-1]          
          shutil.rmtree(os.path.join(r,item)) 
        # end for 
        break
      # end for
    # end for
    
    # Remove builds
    src_path = os.path.join(getpath(),"static/")
    for r, d, f in os.walk(src_path):        
      for item in f:
        ext = item.split('.')[-1]
        if '__init__' not in item:
          os.remove( os.path.join(r,item) )
        # end if 
      # end for 
      for item in d:      
        shutil.rmtree(os.path.join(r,item)) 
      # end for 
      break
    # end for

    self.process_headers_static( base_dir = base_dir, tab=tab)
    self.process_static_include_files( base_dir = base_dir)

  #--------------------------------------------------------------------------------------------------- 



def comment_replacer(match): 
  start,mid,end = match.group(1,2,3) 
  if mid is None: 
    # single line comment 
    return '' 
  elif start is not None or end is not None: 
    # multi line comment at start or end of a line 
    return '' 
  elif '\n' in mid: 
    # multi line comment with line break 
    return '\n' 
  else: 
    # multi line comment without line break 
    return ' ' 

comment_re = re.compile( 
  r'(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?', 
  re.DOTALL | re.MULTILINE 
)


def remove_annotations(string):
  str_nocomm=comment_re.sub(comment_replacer, string)
  str_nocomm= re.sub(r'(?m)^ *#.*\n?', '', str_nocomm)
  for i in range(10):
    str_nocomm = str_nocomm.replace("\n\n","\n")

  return str_nocomm

