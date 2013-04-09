!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
! These routines are available for general use. I ask that you send me
! interesting alterations that are available for public use; and that you
! include a note indicating the original author --  John S. Urban
! Last updated Dec 20, 2008
!=======================================================================--------
! :: kracken        ! define command and default parameter values
! :: rget           ! fetch real    value of name VERB_NAME from the language dictionary
! :: iget           ! fetch integer value of name VERB_NAME from the language dictionary
! :: lget           ! fetch logical value of name VERB_NAME from the language dictionary
! :: sget           ! fetch string  value of name VERB_NAME from the language dictionary.
! :: retrev         ! retrieve token value from Language Dictionary when given NAME
! :: string_to_real ! returns real value from numeric character string NOT USING CALCULATOR
! :: delim          ! parse a string and store tokens into an array
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
module M_kracken_dictionary

! @(#) common length of verbs and entries in Language dictionary
! NOTE:   many parameters were reduced in size so as to just accomodate
!         being used as a command line parser. In particular, some might
!         want to change:
!          ic=30          ! number of entries in language dictionary
!          IPvalue=255    ! ilength of verb value

      implicit none

      integer, parameter,public :: IPverb=20                          ! ilength of verb
      integer, parameter,public :: IPvalue=255                        ! ilength of verb value
      integer, parameter,public :: ic=30                              ! number of entries in language dictionary
      integer, parameter,public :: k_int = SELECTED_INT_KIND(9)       ! integer*4
      integer, parameter,public :: k_dbl = SELECTED_REAL_KIND(15,300) ! real*8
      !=================================================================--------
      ! dictionary for Language routines
      character (len=IPvalue),dimension(ic),public :: values=" " ! contains the values of string variables
      character (len=IPverb),dimension(ic),public  ::    ix2=" " ! string variable names
      integer(kind=k_int),dimension(ic),public :: ivalue=0       ! significant lengths of string variable values
      !================================================================---------
end module M_kracken_dictionary

module M_kracken
   implicit none
   private

   ! SUBROUTINES:
   public :: retrev            ! retrieve token value from Language Dictionary when given NAME
   public :: string_to_real    ! returns real value from numeric character string NOT USING CALCULATOR
   public :: kracken           ! define command and default parameter values
   public :: delim             ! parse a string and store tokens into an array

   private :: parse_two        ! convenient call to parse() -- define defaults, then process user input
   private :: parse            ! parse user command and store tokens into Language Dictionary
   private :: store            ! replace dictionary name's value (if allow=add add name if necessary)
   private :: bounce           ! find location (index) in Language Dictionary where VARNAME can be found
   private :: add_string       ! Add new string name to Language Library dictionary
   private :: send_message
   private :: get_command_arguments ! get_command_arguments: return all command arguments as a string

   ! FUNCTIONS:
   public :: rget    ! fetch real    value of name VERB_NAME from the language dictionary
   public :: iget    ! fetch integer value of name VERB_NAME from the language dictionary
   public :: lget    ! fetch logical value of name VERB_NAME from the language dictionary
   public :: sget    ! fetch string  value of name VERB_NAME from the language dictionary.

   private :: igets  ! return the subscript value of a string when given it's name
   private :: uppers ! uppers: return copy of string converted to uppercase


contains
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
subroutine retrev(name,val,len,ier)
!     Copyright(c) 1989 John S. Urban   all rights reserved
!@(#) retrieve token value from Language Dictionary when given NAME

      use M_kracken_dictionary ! dictionary for Language routines


      character(len=*),intent(in)  ::  name
      character(len=*),intent(out) ::  val
      integer,intent(out)          ::  len
      integer,intent(out)          ::  ier

      integer          ::  isub

      isub=igets(name)  ! get index entry is stored at

      if(isub > 0)then ! entry was in dictionary
         val=values(isub)
         len=ivalue(isub)
         ier=0
      else              ! entry was not in dictionary
         ier=-1
         val=" "
         len=0
      endif

      return

end subroutine retrev
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
subroutine string_to_real(chars,valu,ierr)
!     @(#) returns real value from numeric character string NOT USING CALCULATOR
!     Copyright(c) 1989 John S. Urban   all rights reserved
!
!     returns a real value from a numeric character string.
!
!  o  works with any g-format input, including integer, real, and
!     exponential.
!
!     if an error occurs in the read, iostat is returned in ierr and
!     value is set to zero.  if no error occurs, ierr=0.
!
      character(len=*),intent(in)  ::  chars
      real,intent(out)             ::  valu
      integer,intent(out)          ::  ierr

      integer, parameter :: k_dbl = SELECTED_REAL_KIND(15,300) ! real*8
      character(len=13)  ::  frmt
      integer            ::  ios
      real(kind=k_dbl)   ::  valu8

      write(unit=frmt,fmt="( ""(bn,g"",i5,"".0)"" )")len(chars)
      ierr=0
      read(unit=chars,fmt=frmt,iostat=ios)valu8

      if (ios /= 0 )then
         valu8=0.0_k_dbl
         call send_message("*string_to_real* - cannot produce number from this string")
         call send_message(chars)
         ierr=ios
      endif

      valu=real(valu8)
      return

end subroutine string_to_real
!=======================================================================
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()(
!=======================================================================
function rget(keyword)
! @(#) given keyword, fetch single real value from the language dictionary (zero on error)

   real                ::  rget

   character(len=*),intent(in)    ::  keyword

   character(len=255)  ::  value
   integer             ::  len
   integer             ::  ier
   real                ::  anumber

   value=" "
   call retrev(keyword, value, len, ier)
   call string_to_real(value(:len), anumber, ier)
   rget = anumber

   return

end function rget
!=======================================================================
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()(
!=======================================================================

function iget(keyword)
! @(#) given keyword, fetch single integer value from the language dictionary (zero on error)

   integer                      ::  iget

   character(len=*),intent(in)  ::  keyword

   character(len=255)           ::  value
   integer                      ::  len
   integer                      ::  ier
   real                         ::  anumber

   call retrev (keyword, value, len, ier)
   call string_to_real (value(:len), anumber, ier)
   iget = int(anumber)

   return

end function iget
!=======================================================================
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()(
!=======================================================================
function lget (keyword)
! @(#) given keyword, fetch single logical value from the language dictionary (zero on error)

   logical                      ::  lget

   character(len=*),intent(in)  ::  keyword

   character(len=255)           ::  value
   integer                      ::  len
   integer                      ::  ier

   call retrev (keyword, value, len, ier)
   value=uppers(value,len)
   if(value(:len)==" ")then
      lget=.true.
   elseif(value(:len)=="#N#")then
      lget=.false.
   elseif(value(:1)=="T")then
      lget=.true.
   elseif(value(:1)=="F")then
      lget=.false.
   elseif(value(:2)==".T")then
      lget=.true.
   elseif(value(:2)==".F")then
      lget=.false.
   else
      call send_message("*lget* bad value for logical for "//keyword(:len_trim(keyword)))
      lget=.false.
   endif

   return

end function lget
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
! These routines are available for general use. I ask that you send me
! interesting alterations that are available for public use; and that you
! include a note indicating the original author --  John S. Urban
!=======================================================================--------
subroutine kracken(verb,string)

!     get the entire command line argument list and pass it and the
!     prototype to parse_two()

      character  (len=*),intent(in)  ::  string
      character  (len=*),intent(in)  ::  verb

      character  (len=1024)          ::  command
      integer :: ilen
      integer :: ier

      call get_command_arguments(command,ilen,ier)
      call parse_two(verb,string,command,ilen)

      return

end subroutine kracken
!=======================================================================--------
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
subroutine parse_two(verb,init,pars,ipars)
!
!@(#) convenient call to parse() -- define defaults, then process user input
!
!   verb   is the name of the command to be reset/defined  and then set
!   init   is a string used to add a new command or to reset an old one.
!          This string is usually hard-set in the program.
!   pars   is a string defining the command options to be set, usually
!          from a user input file
!   ipars  is the length of the user-input string pars.

      character(len=*),intent(in)  ::  verb
      character(len=*),intent(in)  ::  init
      character(len=*),intent(in)  ::  pars
      integer,intent(in)           ::  ipars

      integer           ::  ipars2

      call parse(verb(:len_trim(verb)),init,"add") ! initialize command

      if(ipars <= 0)then
         ipars2=len(pars)
      else
         ipars2=ipars
      endif

      call parse(verb,pars(:ipars2),"no_add") ! process user command options

      return

end subroutine parse_two
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
subroutine parse(verb,string,allow)
!     Copyright(c) 1989 John S. Urban   all rights reserved
!!!   need to handle a minus followed by a blank character
!!!   set up odd for future expansion
!
!@(#) parse user command and store tokens into Language Dictionary
!
!     given a string of form
!
!     value  -var value -var value
!     try to define a bunch of variables of form
!     verb_var(i) = value
!
!     values may be in double quotes if they contain -alphameric, a #
!     signifies rest of line is a comment, adjacent double quotes put
!     one double quote into value, processing ends when an unquoted
!     semi-colon or end of string is encountered.
!     the variable name for the first value is verb_init (often verb_oo)
!     call it once to give defaults
!     call it again and vars without values are set to null strings
!     leading and trailing blanks are removed from values
!
!     string is character input string
!
!     if ileave is 0, leave double quotes where you find them; else if 1
!     remove them. Normally, they should be removed
      use M_kracken_dictionary
!=========================================================================
! @(#) for left-over command string for Language routines
!     optionally needed if you are going to allow multiple commands on a line
      ! number of characters left over,
      ! number of non-blank characters in actual parameter list
!=========================================================================

      character(len=*),intent(in)          ::  verb
      character(len=*),intent(in)          ::  string
      character(len=*),intent(in)          ::  allow

      character(len=IPvalue+2)             ::  dummy
      character(len=IPvalue),dimension(2)  ::  var
      character(len=3)                     ::  delmt
      character(len=2)                     ::  init
      character(len=1)                     ::  currnt
      character(len=1)                     ::  prev
      character(len=1)                     ::  forwrd
      character(len=IPvalue)               ::  val
      character(len=IPverb)                ::  name
      integer,dimension(2)                 ::  ipnt
      integer,save                         ::  ileave=1
      integer                              ::  ilist
      integer                              ::  ier
      integer                              ::  islen
      integer                              ::  ipln
      integer                              ::  ipoint
      integer                              ::  itype
      integer                              ::  ifwd
      integer                              ::  ibegin
      integer                              ::  iend

      ilist=1
      init="oo"
      ier=0
      islen=len_trim(string)   ! find number of characters in input string
      ! if input string is blank, even default variable will not be changed
      if(islen  ==  0)then
         return
      endif
      dummy=string
      ipln=len_trim(verb)      ! find number of characters in verb prefix string
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
      if(verb(:ipln)=="MODE")then
         if(string=="LEAVEQUOTES")then
            if(allow=="YES")then
               ileave=0
            elseif(allow=="NO")then
               ileave=1
            else
               call send_message("*parse* LEAVECODES value bad")
               ileave=1
            endif
         else
            call send_message("*parse* UNKNOWN MODE")
         endif
         return
      endif
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
      var(2)=init         ! initial variable name
      var(1)=" "          ! initial value of a string
      ipoint=0            ! ipoint is the current character pointer for (dummy)
      ipnt(2)=2           ! pointer to position in parameter name
      ipnt(1)=1           ! pointer to position in parameter value
      itype=1             ! itype=1 for value, itype=2 for variable
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
      delmt="off"
      prev=" "
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
      do
      ipoint=ipoint+1               ! move current character pointer forward
      currnt=dummy(ipoint:ipoint)   ! store current character into currnt
      ifwd=min(ipoint+1,islen)
      forwrd=dummy(ifwd:ifwd)       ! next character (or duplicate if last)
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
      if((currnt=="-".and.prev==" ".and.delmt == "off".and.index("0123456789.",forwrd) == 0).or.ipoint > islen)then
      ! beginning of a parameter name
         if(ipnt(1)-1 >= 1)then
            ibegin=1
            iend=len_trim(var(1)(:ipnt(1)-1))

            do
               if(iend  ==  0)then   !len_trim returned 0, parameter value is blank
                  iend=ibegin
                  exit
               else if(var(1)(ibegin:ibegin) == " ")then
                  ibegin=ibegin+1
               else
                  exit
               endif
            enddo

            name=verb(:ipln)//"_"//var(2)(:ipnt(2))
            val=var(1)(ibegin:iend)
            call store(name,val,allow,ier)       ! store name and it's value
         else
            name=verb(:ipln)//"_"//var(2)(:ipnt(2))
            val=" "                                 ! store name and null value
            call store(name,val,allow,ier)
         endif
         ilist=ilist+ipln+1+ipnt(2)
         ilist=ilist+1
         itype=2                          ! change to filling a variable name
         var(1)=" "                       ! clear value for this variable
         var(2)=" "                       ! clear variable name
         ipnt(1)=1                        ! restart variable value
         ipnt(2)=1                        ! restart variable name
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
      elseif(currnt == "#".and.delmt == "off")then   ! rest of line is comment
         islen=ipoint
         dummy=" "
         prev=" "
         cycle
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
      ! rest of line is another command(s)
         islen=ipoint
         dummy=" "
         prev=" "
         cycle
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
      else       ! currnt is not one of the special characters
         ! the space after a keyword before the value
         if(currnt == " ".and.itype  ==  2)then
            ! switch from building a keyword string to building a value string
            itype=1
         ! beginning of a delimited parameter value
         elseif(currnt  ==  """".and.itype  ==  1)then
            ! second of a double quote, put quote in
            if(prev  ==  """")then
                var(itype)(ipnt(itype):ipnt(itype))=currnt
                ipnt(itype)=ipnt(itype)+1
                delmt="on"
            elseif(delmt  ==  "on")then     ! first quote of a delimited string
                delmt="off"
            else
                delmt="on"
            endif
            if(ileave  ==  0.and.prev /= """")then  ! leave quotes where found them
               var(itype)(ipnt(itype):ipnt(itype))=currnt
               ipnt(itype)=ipnt(itype)+1
            endif
         else     ! add character to current parameter name or parameter value
            var(itype)(ipnt(itype):ipnt(itype))=currnt
            ipnt(itype)=ipnt(itype)+1
            if(currnt /= " ")then
            endif
         endif
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
      endif
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
      prev=currnt

      if(ipoint <= islen)then
         cycle
      endif
      exit
      enddo

      return
end subroutine parse
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
subroutine store(name1,value1,allow1,ier)
!     Copyright(c) 1989 John S. Urban   all rights reserved
!
!@(#) replace dictionary name's value (if allow=add add name if necessary)

      use M_kracken_dictionary

      character(len=*),intent(in)        ::  name1
      character(len=*),intent(in)        ::  value1
      character(len=*),intent(in)        ::  allow1
      integer,intent(out)                ::  ier

      character(len=IPverb)   ::  name
      integer                 ::  indx
      character(len=10)       ::  allow
      character(len=IPvalue)  ::  value
      character(len=IPvalue)  ::  mssge   !  the  message/error/string  value
      integer                 ::  nlen
      integer                 ::  new
      integer                 ::  ii
      integer                 ::  i10

      name=name1
      value=value1
      allow=allow1
      nlen=len(name1)
      ! determine storage placement of the variable and whether it is new
      call bounce(name,indx,ix2,ier,mssge)
      if(ier  ==  -1)then
         call send_message("error occurred in *store*")
         call send_message(mssge)
         return
      endif
      if(indx > 0)then
!        found the variable name
         new=1
      ! check if the name needs added or is already defined
      else if(indx <= 0.and.allow  ==  "add")then
         ! adding the new variable name in the variable name array
         call add_string(name,nlen,indx,ier)
         if(ier  ==  -1)then
            call send_message("*store* could not add "//name(:nlen))
            call send_message(mssge)
            return
         endif
         new=0
      else
!        did not find variable name but not allowed to add it
         !call send_message("could not find "//name)
         call send_message("E-R-R-O-R: UNKNOWN OPTION "//name)
         ii=index(name,"_")
         if(ii > 0)then
            call send_message(name(:ii-1)//" parameters are")
            do i10=1,ic
               if(name(:ii)  ==  ix2(i10)(:ii))then
                  call send_message(" -"//ix2(i10)(ii+1:len_trim(ix2(i10)))//" "//values(i10)(:ivalue(i10)))
               endif
            enddo
         endif
         return
      endif
      ! ignore special value that means leave alone, used by 'set up' calls to
      ! leave a value alone
      ! note that this will prevent the keyword from being defined.
      if(value(1:4)  ==  "@LV@")then
         ! a new leave-alone flag (for use by a 'defining' call)
         if(new  ==  0) then
            value=value(5:)       ! trim off the leading @LV@
            values(iabs(indx))=value    ! store a defined variable's value
            ivalue(iabs(indx))=len_trim(value)  ! store ilength of string
         endif
      else
         values(iabs(indx))=value            ! store a defined variable's value
         ivalue(iabs(indx))=len_trim(value)     ! store ilength of string
      endif
      return
end subroutine store
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
subroutine bounce(varnam,index,ixn,ier,mssge)
!     Copyright(C) 1989 John S. Urban   All rights reserved
!
!@(#) find location (index) in Language Dictionary where VARNAME can be found
!     (Assuming an alphabetized array of character strings)
!
!     If it is not found report where it
!     should be placed as a NEGATIVE index number.
!
!     It is assumed all variable names are lexically greater
!     than a blank string.

      use M_kracken_dictionary

      character(len=*),intent(in)                     ::  varnam
      integer,intent(out)                             ::  index
      !character(len=IPverb),dimension(ic),intent(in)  ::  ixn
      character(len=*),dimension(:),intent(in)        ::  ixn
      integer,intent(out)                             ::  ier
      character(len=*),intent(out)                    ::  mssge

      integer                              ::  maxtry
      integer                              ::  imin
      integer                              ::  imax
      integer                              ::  i10

      maxtry=int(log(float(ic))/log(2.0)+1.0)
      index=(ic+1)/2
      imin=1
      imax=ic
      do i10=1,maxtry

         if(varnam  ==  ixn(index))then
            return
         else if(varnam > ixn(index))then
            imax=index-1
         else
            imin=index+1
         endif

         if(imin > imax)then
            index=-imin

            if(iabs(index) > ic)then
               mssge="error 03 in bounce"
               ier=-1
               return
            endif

            return

         endif

         index=(imax+imin)/2

         if(index > ic.or.index <= 0)then
            mssge="error 01 in bounce"
            ier=-1
            return
         endif

      enddo

      mssge="error 02 in bounce"

      return

end subroutine bounce
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
subroutine add_string(newnam,nchars,index,ier)
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
!@(#) Add new string name to Language Library dictionary

      use M_kracken_dictionary

!     maximum number of string variables to be stored
      character(len=*),intent(in)       ::  newnam
      integer,intent(in)                ::  nchars
      integer,intent(in)                ::  index
      integer,intent(out)               ::  ier

      integer                ::  istart
      integer                ::  i10

!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
!     if last position in the name array has already been used, then
!     report that no room is left and set error flag and error message.

      if(ix2(ic) /= " ")then
        call send_message("*add_string* no room left to add more string variable names")
        ier=-1

        return

      endif
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
      istart=iabs(index)

!     watch out when ic approaches istart that logic is correct.
      do i10=ic-1,istart,-1
!        pull down the array to make room for new value
         values(i10+1)=values(i10)
         ivalue(i10+1)=ivalue(i10)
         ix2(i10+1)=ix2(i10)
      enddo

      values(istart)=" "
      ivalue(istart)= 0
      ix2(istart)=newnam(1:nchars)

      return

end subroutine add_string
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
function igets(chars0)
!     Copyright(c) 1989 John S. Urban   all rights reserved
!@(#) return the subscript value of a string when given it's name
!     WARNING: only request value of names known to exist

      use M_kracken_dictionary ! dictionary for Language routines

      character(len=*),intent(in)        ::  chars0

      character(len=IPvalue)             ::  msg
      character(len=IPverb)              ::  chars
      character(len=IPvalue)             ::  mssge
      integer                            ::  ierr
      integer                            ::  index
      integer                            ::  igets

      chars=chars0
      ierr=0
      index=0
      call bounce(chars,index,ix2,ierr,mssge) ! look up position

      if((ierr  ==  -1).or.(index <= 0))then
         msg="*igets* variable "//chars//" undefined"
         call send_message(msg)
!!!!!!   very unfriendly subscript value
         igets=-1
      else
         igets=index
      endif

      return

end function igets
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
subroutine delim(line0,array,n,iicount,ibegin,iterm,ilen,dlim)
!     @(#) parse a string and store tokens into an array
!
!     given a line of structure " par1 par2 par3 ... parn "
!     store each par(n) into a separate variable in array.
!
!     IF ARRAY(1) = '#NULL#' do not store into string array  (KLUDGE))
!
!     also icount number of elements of array initialized, and
!     return beginning and ending positions for each element.
!     also return position of last non-blank character (even if more
!     than n elements were found).
!
!     no quoting of delimiter is allowed
!     no checking for more than n parameters, if any more they are ignored
!
      character(len=*),intent(in)                ::  line0
      integer,intent(in)                         ::  n
      !character(len=*),dimension(n),intent(out)  ::  array
      character(len=*),dimension(:),intent(out)  ::  array
      integer,intent(out)                        ::  iicount
      !integer,dimension(n),intent(out)           ::  ibegin
      integer,dimension(:),intent(out)           ::  ibegin
      !integer,dimension(n),intent(out)           ::  iterm
      integer,dimension(:),intent(out)           ::  iterm
      integer,intent(out)                        ::  ilen
      character(len=*),intent(in)                ::  dlim

      character(len=1044)            ::  line
      logical                        ::  lstore
      integer                        ::  idlim
      integer                        ::  icol
      integer                        ::  iarray
      integer                        ::  istart
      integer                        ::  iend
      integer                        ::  i10
      integer                        ::  ifound

      iicount=0
      ilen=len_trim(line0)

      if(ilen > 1044)then
         call send_message("*delim* input line too long")
      endif

      line=line0
      idlim=len(dlim)

      if(idlim > 5)then
         idlim=len_trim(dlim)      ! dlim a lot of blanks on some machines if dlim is a big string
         if(idlim  ==  0)then
            idlim=1  ! blank string
         endif
      endif

!     command was totally blank
      if(ilen  ==  0)then
         return
      endif
!
!     there is at least one non-blank character in the command
!     ilen is the column position of the last non-blank character
!     find next non-delimiter
      icol=1

      if(array(1)  ==  "#NULL#")then    ! special flag to not store into character array
         lstore=.false.
      else
         lstore=.true.
      endif

      do iarray=1,n,1             ! store into each array element until done or too many words
         if(index(dlim(1:idlim),line(icol:icol))  ==  0)then ! if current character is not a delimiter
           istart=icol           ! start new token on the non-delimiter character
           ibegin(iarray)=icol
           iend=ilen-istart+1+1  ! assume no delimiters so put past end of line

           do i10=1,idlim
              ifound=index(line(istart:ilen),dlim(i10:i10))
              if(ifound > 0)then
                iend=min(iend,ifound)
              endif
           enddo

            if(iend <= 0)then                              ! no remaining delimiters
              iterm(iarray)=ilen
              if(lstore)then
                 array(iarray)=line(istart:ilen)
              endif
              iicount=iarray
              return
            else
              iend=iend+istart-2
              iterm(iarray)=iend
              if(lstore)then
                 array(iarray)=line(istart:iend)
              endif
            endif
           icol=iend+2
         else
           icol=icol+1
           cycle
         endif
   !     last character in line was a delimiter, so no text left
   !     (should not happen where blank=delimiter)
         if(icol > ilen)then
           iicount=iarray
           return
         endif
      enddo

!     more than n elements
      iicount=n

      return

end subroutine delim
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
subroutine send_message(msg) ! general message routine
!     Copyright(c) 1989 John S. Urban   all rights reserved
!
!     SIMPLIFIED FOR EXAMPLE: JUST ECHOES MESSAGES
!
      character(len=*),intent(in) :: msg

      print "("" #kracken>:"",a)", msg(:len_trim(msg)) ! echo mode

      return

end subroutine send_message
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
! currently, get_command may or may not contain the command name as well as the
! arguments, and some systems allow blank spaces or other characters that can
! confuse. This routine returns all the arguments as a string.

subroutine get_command_arguments(string,istring_len,istatus)
!     @(#)get_command_arguments: return all command arguments as a string

   character(len=*),intent(out) :: string      !  string of all arguments
   integer,intent(out)          :: istring_len !  last character position set
   integer,intent(out)          :: istatus     !  status (non-zero means error)

   integer                      :: ilength     !  length of individual arguments
   integer                      :: i           !  loop count
   integer                      :: icount      !  count of number of arguments available
   character(len=255)           :: value       !  store individual arguments one at a time

   string=""       ! initialize returned output string
   istring_len=0   ! initialize returned output string length
   istatus=0       ! initialize returned error code

   icount=command_argument_count() ! intrinsic gets number of arguments

   if(icount>0)then  ! if there are arguments load them into string
      ! start with first argument
      call get_command_argument(1,string,istring_len,istatus)

      if(istatus  ==  0)then
         do i=2,icount  ! append any additional arguments to first
            call get_command_argument(i,value,ilength,istatus)
            if(istatus /= 0)then
               exit  ! stop on error
            endif
            string=string(:istring_len)//" "//value(:ilength)
            istring_len=istring_len+ilength+1
         enddo
      endif

      ! keep track of length and so do not need to use len_trim
      istring_len=len_trim(string)
   endif

   return

end subroutine get_command_arguments
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
function uppers(linei,ilen) result (string)
!     @(#)uppers: return copy of string converted to uppercase
!     Copyright 1996 (c), John S. Urban

! put back in if length of input longer than length of output

      character(len=*),intent(in) :: linei
      integer,intent(in) :: ilen

      character(len=ilen) :: string

      character(len=1) :: let
      integer ::  ilet
      integer ::  iout
      integer ::  i10

      iout=1
      string=" "

      do i10=1,ilen,1
         let=linei(i10:i10)
         ilet=ichar(let)
         ! lowercase a-z in ASCII is 97 to 122
         ! uppercase a-z in ASCII is 65 to 90

         if( (ilet >= 97) .and. (ilet <= 122))then
            ! convert lowercase a-z to uppercase a-z
            string(iout:iout)=char(ilet-32)
         else
            ! character is not an uppercase a-z, just put it in output
            string(iout:iout)=let
         endif

         iout=iout+1
      enddo

      return

end function uppers
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
function sget(name,ilen) result (string)
!@(#) Fetch string value of specified NAME from the language dictionary.

!     Copyright(C) 1989 John S. Urban   all rights reserved
!
!     This routine trusts that the desired name exists. A blank
!     is returned if the name is not in the dictionary

      use M_kracken_dictionary ! dictionary for Language routines

      character(len=*),intent(in)  ::  name    !  name to look up in dictionary
      integer,intent(in)           ::  ilen    !  length of returned output string
      character(len=ilen)          ::  string
      integer                      ::  isub

      isub=igets(name) ! given name return index name is stored at

      if(isub > 0)then ! if index is valid return string
         string=values(isub)
      else              ! if index is not valid return blank string
         string(:)=" "
      endif

      return

end function sget
!=======================================================================--------
!()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()
!=======================================================================--------
end module M_kracken