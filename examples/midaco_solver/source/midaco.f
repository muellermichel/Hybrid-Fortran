CCCCCCCCCCCCCCCCCCCCCCCC MIDACO FORTRAN HEADER CCCCCCCCCCCCCCCCCCCCCCCCC
C
C     _|      _|  _|_|_|  _|_|_|      _|_|      _|_|_|    _|_|
C     _|_|  _|_|    _|    _|    _|  _|    _|  _|        _|    _|
C     _|  _|  _|    _|    _|    _|  _|_|_|_|  _|        _|    _|
C     _|      _|    _|    _|    _|  _|    _|  _|        _|    _|
C     _|      _|  _|_|_|  _|_|_|    _|    _|    _|_|_|    _|_|
C
C                                          Version 4.0 (limited)
C
C    MIDACO - Mixed Integer Distributed Ant Colony Optimization
C    ----------------------------------------------------------
C
C    MIDACO solves the general Mixed Integer Non-Linear Program (MINLP):
C
C
C       Minimize     F(X)           where X(1,...N-NI)   is *CONTINUOUS*
C                                   and   X(N-NI+1,...N) is *DISCRETE*
C
C       Subject to:  G_j(X)  =  0   (j=1,...ME)     Equality Constraints
C                    G_j(X) >=  0   (j=ME+1,...M) Inequality Constraints
C
C       And bounds:  XL <= X <= XU
C
C
C    MIDACO is a (heuristic) global optimization solver that stochastically
C    approximates a solution 'X' to the above displayed MINLP problem. MIDACO
C    is based on an extended Ant Colony Optimization framework (see [1]) in
C    combination with the Oracle Penalty Method (see [2]) for constraints 'G(X)'.
C
C    In case of mixed integer problems, the continuous variables are stored
C    first in 'X(1,...N-NI)', while the discrete (also called integer or
C    combinatorial) variables are stored last in 'X(N-NI+1,...N)'.
C    As an example consider:
C
C       X = (  0.1234,  5.6789,  1.0000,  2.0000,  3.0000)
C
C       where 'N' = 5 and 'NI' = 3
C
C    Note that all 'X' is of type double precision. Equality and inequality
C    constraints are handled in a similar way. The vector 'G' stores at first
C    the 'ME' equality constraints and behind those, the remaining 'M-ME'
C    inequality constraints are stored.
C
C    MIDACO is a derivative free black box solver and does not require the
C    relaxation of integer variables (this means, integer variables are
C    treated as categorical variables). MIDACO does not require any user
C    specified parameter tuning as it can run completely on 'Autopilot'
C    (all parameter set equal to zero). Optionally, the user can adjust
C    the MIDACO performance by setting some parameters explained below.
C
C
C    List of MIDACO subroutine arguments:
C    ------------------------------------
C
C    P  :   (Parallelization Factor)
C            If no parallelization is desired, set P = 1.
C
C    N  :   Number of optimization variables in total (continuous and integer ones).
C           'N' is the dimension of the iterate 'X' with X = (X_1,...,X_N).
C
C    NI :   Number of integer optimization variables. 'NI' <= 'N'.
C           Integer (discrete) variables must be stored at the end of 'X'.
C
C    M  :   Number of constraints in total (equality and inequality ones).
C           'P*M' is the dimension of a constraint vector 'G' with G = (G_1,...,G_M).
C
C    ME :   Number of equality constraints. 'ME' <= 'M'.
C           Equality constraints are stored in the beginning of 'G'.
C           Inequality constraints are stored in the end of 'G'.
c
C    X(P*N) :   Array containing the iterate 'X'. For P=1 (no parallelization)
C               'X' stores only one iterate and has length 'N'. For P>1
C               'X' contains several iterates, which are stored one after
C               another.
C
C    F(P)   :   Array containing the objective function value 'F' corresponding
C               to the iterates 'X'. For P=1 (no parallelization), 'F' is a single
C               value. For P>1 F stores several values, corresponding to 'X' one
C               after another.
C
C    G(P*M) :   Array containing the constraint values 'G'.For P=1 (no parallelization)
C               'G' has length 'M'. For P>1 'G' has length 'P*M' and stores the vectors
C               of constraints, corresponding to 'X' one after another.
C
C     XL(N) :   Array containing the lower bounds for the iterates 'X'.
C               Note that for integer dimensions (i > N-NI) the bounds should also be
C               discrete, e.g. X(i) = 1.0000.
C
C     XU(N) :   Array containing the upper bounds for the iterates 'X'.
C               Note that for integer dimensions (i > N-NI) the bounds should also be
C               discrete, e.g. X(i) = 1.0000.
C
C    IFLAG :    Information flag used by MIDACO. Initially MIDACO must be called with IFLAG=0.
C               If MIDACO works correctly, IFLAG flags lower than 0 are used for internal
C               communication. If MIDACO stops (either by submitting ISTOP=1 or automatically
C               by the FSTOP or AUTOSTOP parameter), an IFLAG value between 1 and 9 is returned
C               as final message. If MIDACO detects at start-up some critical problem setup, a
C               ***WARNING*** message is returned by IFLAG as value between 10 and 99. If
C               MIDACO detects an ***MIDACO INPUT ERROR***, an IFLAG value between 100 and 999
C               is returned and MIDACO stops. The individual IFLAG flags are as follows:
C
C               Final Messages:
C               ---------------
C               IFLAG = 1 : Feasible solution,   MIDACO was stopped by MAXEVAL or MAXTIME
C               IFLAG = 2 : Infeasible solution, MIDACO was stopped by MAXEVAL or MAXTIME
C               IFLAG = 3 : Feasible solution,   MIDACO stopped automatically by AUTOSTOP
C               IFLAG = 4 : Infeasible solution, MIDACO stopped automatically by AUTOSTOP
C               IFLAG = 5 : Feasible solution,   MIDACO stopped automatically by FSTOP
C
C               WARNING - Flags:
C               ----------------
C               IFLAG = 51 : Some X(i)  is greater/lower than +/- 1.0D+8 (try to avoid huge values!)
C               IFLAG = 52 : Some XL(i) is greater/lower than +/- 1.0D+8 (try to avoid huge values!)
C               IFLAG = 53 : Some XU(i) is greater/lower than +/- 1.0D+8 (try to avoid huge values!)
C
C               IFLAG = 61 : Some X(i)  should be discrete (e.g. 1.0), but is continuous (e.g. 1.234)
C               IFLAG = 62 : Some XL(i) should be discrete (e.g. 1.0), but is continuous (e.g. 1.234)
C               IFLAG = 63 : Some XU(i) should be discrete (e.g. 1.0), but is continuous (e.g. 1.234)
C
C               IFLAG = 71 : Some XL(i) = XU(i) (fixed variable)
C
C               IFLAG = 81 : F(X) has value NaN for starting point X
C               IFLAG = 82 : Some G(X) has value NaN for starting point X
C
C               IFLAG = 91 : FSTOP is greater/lower than +/- 1.0D+8
C               IFLAG = 92 : ORACLE is greater/lower than +/- 1.0D+8
C
C               ERROR - Flags:
C               --------------
C               IFLAG = 101 :   P   <= 0   or   P  > 1.0D+6
C               IFLAG = 102 :   N   <= 0   or   N  > 1.0D+6
C               IFLAG = 103 :   NI  <  0
C               IFLAG = 104 :   NI  >  N
C               IFLAG = 105 :   M   <  0   or   M  > 1.0D+6
C               IFLAG = 106 :   ME  <  0
C               IFLAG = 107 :   ME  >  M
C
C               IFLAG = 201 :   some X(i)  has type NaN
C               IFLAG = 202 :   some XL(i) has type NaN
C               IFLAG = 203 :   some XU(i) has type NaN
C               IFLAG = 204 :   some X(i) < XL(i)
C               IFLAG = 205 :   some X(i) > XU(i)
C               IFLAG = 206 :   some XL(i) > XU(i)
C
C               IFLAG = 301 :   PARAM(1) < 0   or   PARAM(1) > 1.0D+6
C               IFLAG = 302 :   PARAM(2) < 0   or   PARAM(2) > 1.0D+12
C               IFLAG = 303 :   PARAM(3) greater/lower than +/- 1.0D+12
C               IFLAG = 304 :   PARAM(4) < 0   or   PARAM(4) > 1.0D+6
C               IFLAG = 305 :   PARAM(5) greater/lower than +/- 1.0D+12
C               IFLAG = 306 :   |PARAM(6)| < 1   or   PARAM(6) > 1.0D+12
C               IFLAG = 307 :   PARAM(7) < 0   or   PARAM(7) > 1.0D+8
C               IFLAG = 308 :   PARAM(8) < 0   or   PARAM(8) > 100
C               IFLAG = 309 :   PARAM(7) < PARAM(8)
C               IFLAG = 310 :   PARAM(7) > 0 but PARAM(8) = 0
C               IFLAG = 311 :   PARAM(8) > 0 but PARAM(7) = 0
C               IFLAG = 312 :   PARAM(9) < 0   or   PARAM(9) > 1000
C               IFLAG = 313 :   Some PARAM(i) has type NaN
C
C               IFLAG = 401 :   ISTOP < 0 or ISTOP > 1
C
C               IFLAG = 501 :   Double precision work space size LRW is too small.
C                               ---> RW must be at least of size LRW = 200*N+2*M+1000
C
C               IFLAG = 601 :   Integer work space size LIW is too small.
C                               ---> IW must be at least of size LIW = 2*N+P+1000
C
C               IFLAG = 701 :   Input check failed! MIDACO must be called initially with IFLAG = 0
C
C               IFLAG = 801 :   P > PMAX (user must increase PMAX in the MIDACO source code)
C               IFLAG = 802 :   P*M+1 > PXM (user must increase PXM in the MIDACO source code)
C
C               IFLAG = 900 :   Invalid or corrupted LICENSE-KEY
C
C               IFLAG = 999 :   N > 4. The free test version is limited up to 4 variables.
C
C    ISTOP :   Communication flag to stop MIDACO. If MIDACO is called with
C              ISTOP = 1, MIDACO returns the best found solution in 'X' with
C              corresponding 'F' and 'G'. As long as MIDACO should continue
C              its search, ISTOP must be equal to 0.
C
C    PARAM() :  Array containing 9 parameters that can be selected by the user to adjust MIDACO.
C               (See the user manual for a more detailed description of individual parameters)
C
C     PARAM(1) :  [ACCURACY] Accuracy for constraint violation (measured as the L1-Norm over G(X)).
C                 If PARAM(1) is set to 0, MIDACO uses a default accuracy of 0.001. If the user
C                 desires a more precise accuracy, set PARAM(1) = 0.000001 for example. If the
C                 user desires a less precise accuracy, set PARAM(1) = 0.05 for example.
C
C     PARAM(2) :  [RANDOM-SEED] Random seed used for MIDACO's internal pseudo random number
C                 generator. Value must be a positive discrete value, e.g. PARAM(2) = 1,2,3,...
C                 The default random seed for MIDACO is zero.
C
C     PARAM(3) :  [FSTOP] Stopping criteria for MIDACO. MIDACO will stop, if a feasible
c                 solution 'X' with F(X) <= FSTOP is found. FSTOP is disabled, when FSTOP = 0.
C                 In case the user wishes to use zero as FSTOP, a tiny value (e.g. 0.000001)
C                 should be used instead.
C
C     PARAM(4) :  [AUTOSTOP] Automatic stopping criteria within MIDACO. AUTOSTOP defines the
C                 number of successive internal MIDACO restarts, after which no progress was
C                 made on the objective function. AUTOSTOP must be a (discrete) value >= 0.
C                 Some (very rough) examples for AUTOSTOP values are:
C                 AUTOSTOP = 1 (lowest chance of global optimality, but fastest runtime)
C                 AUTOSTOP = 10 (medium chance of global optimality)
C                 AUTOSTOP = 50 (high chance of global optimality)
C                 AUTOSTOP = 500 (very high chance of global optimality, but very long runtime)
C                 AUTOSTOP = 0 disables this stopping criteria.
C
C     PARAM(5) :  [ORACLE] This parameter affects only constrained problems. If PARAM(4)=0.0,
C                 MIDACO will use its internal oracle strategy. If PARAM(4) is not equal to 0.0,
C                 MIDACO will use PARAM(5) as initial oracle for its in-build oracle penalty
C                 function. In case the user wishes to use zero as ORACLE, a tiny value
C                 (e.g. 0.000001) should be used as ORACLE. Note: For most problems it is
C                 better to overestimate an oracle, instead of underestimating it.
C
C     PARAM(6) :  [FOCUS] This parameter focuses the MIDACO search process around the
C                 current best solution (which might be the starting point).
C
C     PARAM(7) :  [ANTS] Number of ants (stochastically sampled iterates) used by MIDACO
C                 in every generation. ANTS must be greater or equal to KERNEL.
C
C     PARAM(8) :  [KERNEL] Solution archive size of MIDACO. KERNEL must be lower or equal to ANTS.
C
C     PARAM(9) :  [CHARACTER] This parameter allows the user to use highly customized MIDACO
C                 internal parameter settings for specific problems. This parameter is currently
C                 only available as a service. In case you are interested in this service, please
C                 contact info@midaco-solver.com.
C
C
C    RW(LRW) :  Real workarray (Type: double precision) of length 'LRW'
C    LRW :      Length of 'RW'. 'LRW' must be greater or equal to  200*N+2*M+1000
C
C    IW(LIW) :  Integer workarray (Type: long integer) of length 'LIW'
C    LIW :      Length of 'IW'. 'LIW' must be greater or equal to  2*N+P+1000
C
C    KEY :  License-Key for MIDACO. Note that any licensed copy of MIDACO comes with an
C           individual 'KEY' determining the license owner and its affiliation.
C
C
C    References:
C    -----------
C
C    [1] Schlueter, M., Egea, J. A., Banga, J. R.:
C        "Extended ant colony optimization for non-convex mixed integer nonlinear programming",
C        Computers & Operations Research (Elsevier), Vol. 36 , Issue 7, Page 2217-2229, 2009.
C
C    [2] Schlueter M., Gerdts M.: "The oracle penalty method",
C        Journal of Global Optimization (Springer), Vol. 47(2),pp 293-325, 2010.
C
C
C    Author (C) :   Dr. Martin Schlueter
C                   Information Initiative Center,
C                   Division of Large Scale Computing Systems,
C                   Hokkaido University, JAPAN.
C
C    Email :        info@midaco-solver.com
C    URL :          http://www.midaco-solver.com
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE MIDACO( P, N, NI, M, ME, X, F, G, XL, XU, IFLAG,
     &                   ISTOP, PARAM, RW, LRW, IW, LIW, KEY)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      INTEGER P, N, NI, M, ME, IFLAG, LRW, IW, LIW, ISTOP
      DOUBLE PRECISION X, F, G, XL, XU, PARAM, RW
      CHARACTER*60 KEY
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      DIMENSION X(P*N),F(P),G(P*M),XL(N),XU(N),RW(LRW),IW(LIW),PARAM(9)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Increase the size of PMAX and PXM if necessary (IFLAG: 801 or 802)
      INTEGER PMAX, PXM
      PARAMETER (PMAX =  1024 * 32)
      PARAMETER (PXM  = 1024 * 32 * 25 + 1)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C     Scrambled Source Code Starts Here
C
      INTEGER I
      DOUBLE PRECISION A(PMAX),B(PMAX),GM(PXM)
      if(iflag.eq.-999) istop = 1
      if(iflag.eq.0)then
c         Check P <= PMAX
          if(P.gt.PMAX)then
              iflag = 801
              return
          endif
c         Check P*M+1 <= PXM
          if(P*M+1.gt.PXM)then
              iflag = 802
              return
          endif
      endif
      if(m.gt.0)then
          do I = 1,P*M
              GM(I) = g(I)
          enddo
      endif
      GM(P*M+1) = 0.0D0
      call jfk(P,n,ni,m,me,x,f,GM,xl,xu,iflag,istop,
     &                 param,rw,lrw,iw,liw,PMAX,A,B,KEY)
      end
                  subroutine i015(f,m,i8,g,i5,i2,i19,i49,
     &                           i4,i32,i6,i99,i42,i93,i36)
            implicit none
      integer f,m,i8,i19,i49,i32,i6,i99,i42,i93,i,j,i10,i43,
     &  i21,i39,i62,i44,i38,i20,i95,i36(*),
     &                        i71
                           double precision g,i5,i2,i4,i02,i35
            dimension g(m),i5(m),i2(m),i4(i32),i6(i99)
         data i10,i43,i21,i39,i62,i38,i20
     &    /0,0,0,0,0,0,0/
                            data i35 /0.0d0/
                                   if(i42.eq.-30)then
                i10 = 0
                                                     i43 = 31 + f + 1
                                                 do i = 1,m
                        j             = int(i*i02(i4(1))) + 1
          i6(i43+i-1) = i6(i43+j-1)
                 i6(i43+j-1) = i
                                                if(i.ge.-25-i42)then
                      do j = 1,i
             i4(j) = dble(i42)-i02(i4(1))
                                enddo
                                                            endif
                                   enddo
                                  i6(31) = 1
                                                 i38 = i43 + m
                                      do i = 1,m
                                        i6(i38+i-1) = 0
                                                         enddo
                                          if(i4(1).ge.0.9d0)then
         i44 = 92+m+f+m
                             i6(i44) = 0
                        do i=1,30
                             i6(i44) = i6(i44) + i6(f+40+m+i+m)
                                         enddo
                       if(1372-i6(i44).ne.0)then
                                     do i=1,i99
                                i4(i) = dble(i6(i))
                      enddo
          i42 = int(i4(1))*1000
                                                            goto 22
                       endif
        endif
                    endif
                                   i95 = 0
                                            if(i6(31).eq.0)then
                    i20 = i6(i43+i10-1)
                           i6(30) = i20
                                i39 = i39 + 1
                    i21 = - i21
                                       i35 = i35 / dble(2**i36(14))
                          if(i35.lt.1.0d0/dble(10*i36(15)))then
                             i35  = 1.0d0/dble(10*i36(15))
        endif
         if(i20.gt.m-i8.and.i39.gt.i62)then
                                            i6(i38+i20-1) = 1
                                                   if(i10.ge.m) goto 2
                                      i95 = 1
           endif
                        i71 = i36(16)
        if(i20.le.m-i8.and.i39.gt.i71)then
                                i6(i38+i20-1) = 1
                                                     if(i10.ge.m) goto 2
                        i95 = 1
                                           endif
                            if(dabs(i5(i20)-i2(i20)).le.1.0d-12)then
                                  i6(i38+i20-1) = 1
                                              if(i10.ge.m) goto 2
           i95 = 1
                     endif
                  endif
                     if(i6(31).eq.1.or.i95.eq.1)then
                                                          i10 = i10 + 1
                          if(i10.gt.m) goto 2
                                  i20 = i6(i43+i10-1)
        i6(30) = i20
                                                     i39 = 1
                              if(i20.gt.m-i8)then
         if(i4(i19+i20-1).eq.i5(i20).or.
     &      i4(i19+i20-1).eq.i2(i20))then
                                                  i62 = 1
          else
                                                i62 = 2
                                           endif
              endif
              if(i02(i4(1)).ge.0.5d0)then
                                                             i21 = 1
                        else
               i21 = -1
                               endif
                                     i35 = dsqrt(i4(i49+i20-1))
                                endif
                             do i = 1,m
                                       g(i) = i4(i19+i-1)
                                      enddo
                     if(i20.le.m-i8)then
               g(i20) = g(i20) + i21 * i35
            else
                                   g(i20) = g(i20) + i21
        if(g(i20).lt.i5(i20))then
                               g(i20) = i5(i20) + 1
                        endif
                               if(g(i20).gt.i2(i20))then
                            g(i20) = i2(i20) - 1
               endif
                      endif
                           if(g(i20).lt.i5(i20)) g(i20) = i5(i20)
             if(g(i20).gt.i2(i20)) g(i20) = i2(i20)
          if(i20.gt.m-i8) g(i20) = dnint(g(i20))
                   if(i10.eq.1.and.i39.eq.1)then
                                                     i42 = -30
                               else
                          i42 = -31
                                            endif
                             return
    2      i42 = -40
            do i = 1,m
                            if(i6(i38+i-1).eq.0) goto 22
                                                       enddo
                  i93 = 1
                         i42 = -99
   22                        return
                                                                   end
                                  subroutine i022(l,x,n)
                                              implicit none
             integer n,i
                    double precision l,x
               dimension x(n)
                                     if(l.ne.l)then
          l = 1.0d16
                                        endif
                   do i=1,n
                                if(x(i).ne.x(i))then
                        x(i) = - 1.0d16
                                                            endif
                                                        enddo
                                                       end
        subroutine i07(m,i8,g,i5,i2,i4,i32,i6,i99,i18,i19,i36)
                                  implicit none
                   integer m,i8,i32,i19,i,i6,i99,i18,i36(*)
                   double precision g,i5,i2,i4,i02,i34,i35,i04
                        dimension g(m),i5(m),i2(m),i4(i32),i6(i99)
                                                         do i = 1,m-i8
                         i34 = i02(i4(1))
                                         if(i34.le.0.25d0)then
             g(i) = i4(i19+i-1)
          else
       i35 = (i2(i)-i5(i)) / dble(i6(i18)**2)
                g(i) = i4(i19+i-1) + i35 *
     &           i04(i02(i4(1)),i02(i4(1)))
                                           endif
                  enddo
                                                  do i = m-i8+1,m
                                if(i02(i4(1)).le.0.75d0)then
                            g(i) = i4(i19+i-1)
                                                  else
                                       if(i02(i4(1)).le.0.5d0)then
                                   g(i) = i4(i19+i-1) + 1.0d0
       else
                   g(i) = i4(i19+i-1) - 1.0d0
           endif
                               endif
                                   if(g(i).lt.i5(i)) g(i) = i5(i)
                 if(g(i).gt.i2(i)) g(i) = i2(i)
                      enddo
                         i34 = i02(i4(1))
                                                         do i = 1,m-i8
                                  if(g(i).lt.i5(i))then
                            if(i34.ge.0.1d0*dble(i36(2)))then
      g(i) = i5(i) + (i5(i)-g(i)) / dble(3**i36(3))
         if(g(i).gt.i2(i)) g(i) = i2(i)
                                                        else
                                                      g(i) = i5(i)
                                    endif
                                                             goto 2
                         endif
                          if(g(i).gt.i2(i))then
                           if(i34.ge.0.1d0*dble(i36(2)))then
              g(i) = i2(i) - (g(i)-i2(i)) / dble(3**i36(3))
               if(g(i).lt.i5(i)) g(i) = i5(i)
                           else
      g(i) = i2(i)
                                          endif
                                                               endif
    2                continue
                                     enddo
                                                  end
           subroutine i05(l,i48,i17,i16,i42)
             implicit none
      double precision l,i48(*),i17,i16
                       integer i42
                        if(i48(3).eq.0.0d0) return
                                           if(l.le.i48(3))then
                                   if(i17.le.i16)then
                                         i42 = 5
                                                    endif
                                                 endif
                                                         end
                                               function i04(a,b)
                  implicit none
                   double precision i04, a, b, g(30), i(30)
                                                             data g /
     &        0.260390399999d0, 0.371464399999d0, 0.459043699999d0,
     &0.534978299999d0, 0.603856999999d0, 0.668047299999d0,
     &    0.728976299999d0, 0.787597599999d0, 0.844600499999d0,
     &            0.900516699999d0, 0.955780799999d0, 1.010767799999d0,
     &   1.065818099999d0, 1.121257099999d0, 1.177410099999d0,
     & 1.234617499999d0, 1.293250299999d0, 1.353728799999d0,
     &   1.416546699999d0, 1.482303899999d0, 1.551755799999d0,
     &      1.625888099999d0, 1.706040699999d0, 1.794122699999d0,
     &          1.893018599999d0, 2.007437799999d0, 2.145966099999d0,
     &       2.327251799999d0, 2.608140199999d0, 2.908140199999d0/
                                    data i /
     &      0.207911799999d0,  0.406736699999d0,  0.587785399999d0,
     &   0.743144899999d0,  0.866025499999d0,  0.951056599999d0,
     & 0.994521999999d0,  0.994521999999d0,  0.951056599999d0,
     &          0.866025499999d0,  0.743144899999d0,  0.587785399999d0,
     &    0.406736699999d0,  0.207911799999d0, -0.016538999999d0,
     & -0.207911799999d0, -0.406736699999d0, -0.587785399999d0,
     &   -0.743144899999d0, -0.866025499999d0, -0.951056599999d0,
     &        -0.994521999999d0, -0.994521999999d0, -0.951056599999d0,
     &  -0.866025499999d0, -0.743144899999d0, -0.587785399999d0,
     &-0.406736699999d0, -0.207911799999d0, -0.107911799999d0/
      i04 = g(max(nint(a*30.0d0),1)) *  i(max(nint(b*30.0d0),1))
             end
                  subroutine i09(m,i4,i32,i27,i66,i45,i16,
     /                 i19,i14,i40,i11,i6,i99,i12,o,i28,i24,i22,
     /                     i68,i48,i36)
                                               implicit none
      integer m,i6,i32,i99,i12,o,i28,i24,i22,i19,i14,i40,
     &                                i11,i27,i66,i45,i,j,i36(*)
                        double precision i4,i16,i68,i48(*),i02
                           dimension i4(i32),i6(i99)
                              integer i96,i74,i73,i72
                                                     data i96 /0/
               if(i6(i12).le.1)then
                                                        i6(10) = 0
          i96 = 0
                                                             else
                                         i96 = i96 + 1
                            i6(10) = i36(4)**i96
       endif
          if(i48(6).lt.0.0d0.and.i6(10).ne.0)then
                                 i6(10) = nint(dabs(i48(6)))
                                                              endif
                                i74 = i36(5)
                                                i73 = 5*i36(1)
         i72 = 2
         i6(i28) = i72 * nint( i02(i4(1)) * dble(m) )
        if(i48(7).ge.2.0d0) i6(i28) = nint(i48(7))
              if(i6(i28).lt.i74) i6(i28) = i74
                     if(i6(i28).gt.i73) i6(i28) = i73
        i6(o) = nint( i02(i4(1)) * dble(i6(i28)) )
                       if(i48(8).ge.2.0d0) i6(o) = nint(i48(8))
                           if(i6(o).lt.2) i6(o) = 2
         i6(i24) = i6(i28)
     &       + i36(6) * nint(i02(i4(1)) * dble(i6(i28)))
                               i6(i22) = nint( 1.5d0 * dble(i6(o)) )
              if(i27-i12.gt.i12**i12) i6(o) = i12-i27
                  i4(i45) = i68
      if(i4(i66).le.i16.and.i4(i27).lt.i68) i4(i45) = i4(i27)
      do j=1,i6(o)
                                                do i=1,m
            i4(i19+(j-1)*m+i-1) = 1.06660d90
                                            enddo
                        i4(i40+j-1)       = 1.07770d90
       i4(i14+j-1)         = 1.08880d90
                                     i4(i11+j-1)         = 1.09990d90
                                   enddo
               end
                 subroutine i01(f,m,i8,n,i0,g,l,x,i5,i2,
     &     i42,i41,i48,i4,i32,i6,i99,
     &                    i30,i52,i50,i100,
     &             i990)
                                          implicit none
                  integer f,m,i8,n,i0,i42,i32,i6,i99,i41
          double precision g,l,x,i5,i2,i48(*),i4
            dimension g( f*m ),l( f ),x( f*n+1 ),i5(m),i2(m)
                                            dimension i4(i32),i6(i99)
                      character*60 i990
                integer i30,i52,i50,i100
                                          integer i
                        if(m.gt.4)then
        i42 = 999
       goto 701
                              endif
                               i30 = 100
                                       if(f.le.0.or.f.gt.10**6)then
                                             i42 = 101
                                                 goto 701
         endif
               if(m.le.0.or.m.gt.10**6)then
        i42 = 102
                        goto 701
                                         endif
                                                    if(i8.lt.0)then
                                 i42 = 103
                                              goto 701
           endif
                                                         if(i8.gt.m)then
                                                i42 = 104
                                       goto 701
          endif
           if(n.lt.0.or.n.gt.10**6)then
                                               i42 = 105
                                                            goto 701
                                      endif
                                                     if(i0.lt.0)then
           i42 = 106
                     goto 701
                       endif
                                      if(i0.gt.n)then
               i42 = 107
                                                    goto 701
                                         endif
                     do i=1,m
                      if(g(i).ne.g(i))then
                i42 = 201
                                                   goto 701
                        endif
                                                 if(i5(i).ne.i5(i))then
         i42 = 202
         goto 701
           endif
              if(i2(i).ne.i2(i))then
                       i42 = 203
                                      goto 701
        endif
                              if(g(i).lt.i5(i)-1.0d-4)then
                  i42 = 204
                                         goto 701
                                 endif
                                           if(g(i).gt.i2(i)+1.0d-4)then
       i42 = 205
               goto 701
               endif
                  if(i5(i).gt.i2(i)+1.0d-4)then
                                         i42 = 206
             goto 701
                                                        endif
                                         enddo
           if(i48(1).lt.0.0d0.or.i48(1).gt.1.0d6)then
                         i42 = 301
               goto 701
                                                  endif
          if(i48(2).lt.0.0d0.or.i48(2).gt.1.0d12)then
                                                  i42 = 302
                                                goto 701
                                 endif
                             if(dabs(i48(3)).gt.1.0d12)then
                                                   i42 = 303
                                            goto 701
                                                           endif
       if(i48(4).lt.0.0d0.or.i48(4).gt.1.0d6)then
                       i42 = 304
                     goto 701
                                             endif
             if(dabs(i48(5)).gt.1.0d12)then
                                                  i42 = 305
                                           goto 701
                             endif
                if(i48(6).ne.0.0d0.and.dabs(i48(6)).lt.1.0d0
     &                  .or.dabs(i48(6)).gt.1.0d12)then
                i42 = 306
                                                               goto 701
                                                  endif
                  if(i48(7).lt.0.0d0.or.i48(7).gt.1.0d8)then
        i42 = 307
           goto 701
                                                  endif
                        if(i48(8).lt.0.0d0.or.i48(8).gt.i30)then
                           i42 = 308
                                             goto 701
                                                            endif
                 if(i48(7).gt.0.0d0.and.i48(7).lt.i48(8))then
                                                          i42 = 309
                  goto 701
                                      endif
                   if(i48(7).gt.0.0d0.and.i48(8).eq.0.0d0)then
            i42 = 310
       goto 701
                                                       endif
                        if(i48(7).eq.0.0d0.and.i48(8).gt.0.0d0)then
                                 i42 = 311
                                                             goto 701
                             endif
                 if(i48(9).lt.0.0d0.or.i48(9).gt.1.0d3)then
                                                      i42 = 312
                                   goto 701
                                            endif
                                                do i=1,9
                            if(i48(i).ne.i48(i))then
                                         i42 = 313
                                            goto 701
                                         endif
                                          enddo
                        if(i41.lt.0.or.i41.gt.1)then
                             i42 = 401
                     goto 701
                   endif
                                         i52 = 2*m+n+(m+5)*i30+8
                       i50 = 31+f+m+m
            if(i32.lt.i52+5+2*m+n)then
                           i42 = 501
                                                          goto 701
                                 endif
                        if(i99.lt.i50 + 61)then
                                                 i42 = 601
                                            goto 701
                                           endif
                                          do i=1,i52+5+m+n
                                             i4(i) = 0.0d0
                                                        enddo
         do i=1,i50
                          i6(i) = 0
                    enddo
                      call i023(i6(i50+1),i990)
                i6(i50+61) = 0
                                                   do i=1,60
                      i6(i50+61) =i6(i50+61) + i6(i50+i)
                          enddo
                       if(i6(i50+61).ne.2736)then
                            i42 = 900
                 goto 701
         endif
                                      i100 = 0
                                                         do i=1,m
              if(g(i).gt.1.0d+8.or.g(i).lt.-1.0d+8)then
                                     i42 = 51
                                                 goto 702
                          endif
              if(i5(i).gt.1.0d+8.or.i5(i).lt.-1.0d+8)then
                              i42 = 52
            goto 702
                                                           endif
               if(i2(i).gt.1.0d+8.or.i2(i).lt.-1.0d+8)then
           i42 = 53
                          goto 702
                                                         endif
                  if(i5(i).eq.i2(i))then
                i42 = 71
                      goto 702
                                                  endif
                 enddo
                                                    do i = m-i8+1,m
                           if(dabs(g(i)-dnint(g(i))).gt.1.0d-4)then
                                        i42 = 61
       goto 702
               endif
                        if(dabs(i5(i)-dnint(i5(i))).gt.1.0d-4)then
                                                          i42 = 62
                                       goto 702
                                                                  endif
           if(dabs(i2(i)-dnint(i2(i))).gt.1.0d-4)then
                                  i42 = 63
                                              goto 702
                              endif
                         enddo
                         if(l(1).ne.l(1))then
             i42 = 81
                                                       goto 702
                             endif
         do i = 1,n
                   if(x(i).ne.x(i))then
                                             i42 = 82
        goto 702
                                                   endif
                                   enddo
                                       if(dabs(i48(3)).gt.1.0d+8)then
               i42 = 91
                         goto 702
                                 endif
                                      if(dabs(i48(5)).gt.1.0d+8)then
                                             i42 = 92
                                                     goto 702
                endif
                             return
  701                     continue
                                                        i41 = 1
                                          return
  702                                continue
                                           i100 = 1
                                          return
                                           end
                          subroutine i023(i67s,i15)
                                   implicit none
                              character*60 i15
                        integer i67s(*),i
                             do i = 1,60
               call alphabet(i15(i:i),i67s(i))
            enddo
                                                          end
          subroutine jfk(f,m,i8,n,i0,g,l,x,i5,i2,
     &                              i42,i41,i48,i4,i32,i6,i99,
     &                                       i26,p,i17,i990)
                                            implicit none
           integer f,m,i8,n,i0,i42,i32,i6,i99,i41
                     double precision g,l,x,i5,i2,i48(*),i4
          dimension g(f*m),l(f),x(f*n+1),i5(m),i2(m),i4(i32),i6(i99)
         character*60 i990
                                                        integer i26
                       double precision p(i26),i17(i26)
                    logical i54
               integer i30,i97,i55,i53,j,i90,
     &    i50,i77,i64,i59,i56,i58,i75,
     &                 i101,i92,i94,i,c,i46,i68,
     &          i52,i63,i37,i100,i980,i36(20)
             double precision i02,i35,i60,i70,i79,i16
                                 double precision i04
                  data i30,i97,i63,i37,i55,i53,i68,
     &        i52,i50,i77,i64,i59,i56,i58,i75,
     &                    i101,i92,i94,i100,
     &      i980,i36
     &           /0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     &            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0/
       data i60,i70,i16  /0.0d0,0.0d0,0.0d0/
                                                   if(i42.ge.0)then
                               i63 = 0
             i37   = 0
                           if(i48(1).le.0.0d0)then
           i16 = 1.0d-3
                                  else
         i16 = i48(1)
                                           endif
          if(i42.gt.10.and.i42.lt.100)then
                                   i42 = -3
                  i100 = 0
                                         goto 79
                            endif
                                            call i024(i36,m,i8)
                                                    i97 = 0
                             call i01(f,m,i8,n,i0,g,l,x,i5,i2,
     &             i42,i41,i48,i4,i32,i6,i99,
     &                                          i30,i52,i50,
     &       i100,
     &      i990)
          if(i42.ge.100) goto 86
                    if(i100.eq.1)then
                                 i980 = i42
                                                               i42 = 0
                                            endif
                                                     i97 = 1
                i46 = int(i48(2))
                          i92 = int(i48(4))
                                   i60    = 1.0d16
                    i70  = 1.0d16
         i59      = 1
                                               i56      = i59 + m
                   i58      = i56 + 1
       i75    = i58 + n
                 i68     = i75 + 1
                                        do i=1,m
                                         i4(i52+i59+i-1) = g(i)
                                enddo
                                        do i=1,n
                                         i4(i52+i58+i-1) = x(i)
                                           enddo
                                         i4(i52+i56) = l(1)
                                 call i014(i4(i52+i75),x,n,i0,i16)
            i77 = 0
                                      i101 = 0
                                             i53 = i46
            if(i56-i59.gt.1+i97*3)then
                      i59      = 2
                 i56      = i59 + m
                                        i58      = i56 + 1
                    i75    = i58 + n
         i68     = i75 + 1
              do i=1,m
                           i4(i52+i59+i-1) = g(i)
                                                  enddo
             do i=1,n
          i4(i52+i58+i-1) = x(i)
                                                               enddo
             i48(6) = -1.0d16
                                                  i60    = 1.0d16
                                                      i70  = 1.0d16
                                                  endif
                       i55  = 0
                               if(i4(i52+i75).gt.i16)then
                                     if(i48(5).eq.0.0d0)then
                   i4(i52+i68) = 1.0d9 + i4(i52+i56)
                                                    else
                                    i4(i52+i68) = i48(5)
                                                     endif
       else
               i4(i52+i68) = i4(i52+i56)
         endif
                           else
                                   if(i97.ne.1)then
         i42 = 701
                                                   i41 = 1
                                     return
           endif
             do c = 1,f
                             call i022(l(c),x((c-1)*n+1),n)
                                                               enddo
                                     endif
   79                                                    continue
        if(i42.eq.-300)then
               i55 = 0
          i53 = i53 + 1
                        endif
                                if(i41.eq.0)then
                                                     i54 = .false.
                                   else
                                       i54 = .true.
       endif
                      call i08(f,m,i8,n,i0,g,l,x,i5,i2,i16,
     &                  i63,i37,i55,i54,i53,
     &   i4,i32,i6,i99,i30,i4(i52+i68),
     &       i48,p,i17,i26,i36)
                                              i42 = i55
      if(i42.eq.5) goto 1
                                   if(i42.eq.801)return
                            if(i54)then
                    if(i4(i52+i75).gt.i16. and .i4(2+m+1+n).lt.
     &                                 i4(i52+i75))    goto 1
                 if(i4(i52+i75).le.i16. and .i4(2+m+1+n).le.i16
     &      .and.i4(2+m).lt.i4(i52+i56)) goto 1
                                           goto 3
                                             endif
           if(i55.eq.-3) i77 = i77 + 1
          if(i32.gt.100.and.i56.gt.i59*5)then
                                          i4(i52+i56)     = i4(2+m)
                i4(i52+i75)   = i4(2+m+1+n)
             do i=1,m
                   i4(i52+i59+i-1) = i4(2+i-1)
                                                       enddo
                        do i=1,n
            i4(i52+i58+i-1) = i4(2+m+1+i-1)
                         enddo
                                  if(i4(i52+i75).le.i16)then
                i4(i52+i68) =  i4(i52+i56)
                 endif
                     l(1) = i4(i52+i56)
                                                       do i = 1,m
                      g(i) = i4(i52+i59+i-1)
                          enddo
                                       do i = 1,n
                   x(i) = i4(i52+i58+i-1)
                                                                enddo
                                                         goto 79
                                                          endif
                         i64 = i36(7)
                                       if(i77.ge.i64)then
        i101 = i101 + 1
                       if(i4(i52+i75).gt.i16. and .i4(2+m+1+n).lt.
     &                                 i4(i52+i75))   goto 11
            if(i4(i52+i75).le.i16. and .i4(2+m+1+n).le.i16.and.
     &       i4(2+m).lt.i4(i52+i56))goto 11
                               goto 12
   11          i4(i52+i56)     = i4(2+m)
                                   i4(i52+i75)   = i4(2+m+1+n)
                                      do i=1,m
                     i4(i52+i59+i-1) = i4(2+i-1)
                                         enddo
                                      do i=1,n
              i4(i52+i58+i-1) = i4(2+m+1+i-1)
              enddo
                                       if(i4(i52+i75).le.i16)then
                 i4(i52+i68) =  i4(i52+i56)
                endif
                                   goto 13
   12  continue
   13                                   do i = 2,i52
              i4(i) = 0.0d0
                                       enddo
                        do i = 1,i50
                                                  i6(i) = 0
                                              enddo
        if(i02(i4(1)).ge.0.1d0*dble(i36(9)).or.i48(6).lt.0.0d0)then
                                           do i=1,m
                               if(m.le.m-i8) i35 = (i2(i)-i5(i))
     &                     / (i02(i4(1))*dble(10**i36(17)))
                            i79 = ((i2(i)-i5(i))-(i2(i)-i5(i))
     &                                         / dsqrt(dble(i8)+0.1d0))
     &        / dble(i36(10))
                       if(m.gt.m-i8) i35 = (i2(i)-i5(i)) / 100
                            if(i.gt.m-i8.and.i35.lt.i79)then
                     i35 = i79
                                 endif
                           if(i48(6).ne.0.0d0)then
                  if(i.le.m-i8)then
                                i35 = (i2(i)-i5(i)) / dabs(i48(6))
                                              else
                     i35 = 1.0d0 / dsqrt(dabs(i48(6)))
                                                     endif
                                              endif
                          g(i) = i4(i52+i59+i-1) + i35 *
     &                         i04(i02(i4(1)),i02(i4(1)))
                                           if(g(i).lt.i5(i))then
                                        g(i)=i5(i)+(i5(i)-g(i)) / 2.0d0
                         endif
                                           if(g(i).gt.i2(i))then
          g(i)=i2(i)-(g(i)-i2(i)) / 2.0d0
                                                                   endif
                                                 if(g(i).lt.i5(i))then
                                                  g(i)=i5(i)
                                  endif
                                                if(g(i).gt.i2(i))then
                g(i)=i2(i)
                       endif
                        if(i.gt.m-i8)then
                                               g(i)=dnint(g(i))
                                   endif
                      if(i.gt.2*(i59*2))then
                                             do j=1,i32
                                 i4(j) = i02(i4(1))
                            enddo
                        do j=1,i99
                i6(j) = nint(i02(i4(1)))
                                                              enddo
                                   endif
                            enddo
                                                      else
                                          do i=1,m
                  g(i) = i5(i) + i02(i4(1)) * (i2(i)-i5(i))
                         if (i.gt.m-i8) g(i) = dnint(g(i))
                                             enddo
                              endif
                        i42 = -300
                          i77 = 0
          if(i92.gt.0)then
                                       if(i60.eq.1.0d16)then
                 i94 = 0
                                      i60   = i4(i52+i56)
                                    i70 = i4(i52+i75)
                                     else
                              if(i4(i52+i75).le.i70)then
                                         if(i70.le.i16)then
                                         if(i4(i52+i56).lt.
     &    i60-dabs(i60/1.0d6))then
                       i60   = i4(i52+i56)
                   i70 = i4(i52+i75)
                                                    i94 = 0
                                              else
                             i94 = i94 + 1
                                                     goto 76
                                                   endif
                                                          else
           i94 = 0
                     i60   = i4(i52+i56)
                  i70 = i4(i52+i75)
                                                     endif
                                 else
               i94 = i94 + 1
                               goto 76
           endif
                                                  endif
   76              continue
                                 if(i94.ge.i92)then
                             if(i4(i52+i75).le.i16)then
                          i42 = 3
                                    else
                                    i42 = 4
                            endif
            goto 3
                                                                  endif
                                  endif
                       endif
                                    if(i100.eq.1)then
                            i42 = i980
                  endif
    4                        return
    1   i4(i52+i56)     = i4(2+m)
                           i4(i52+i75)   = i4(2+m+1+n)
           do i=1,m
                        i4(i52+i59+i-1) = i4(2+i-1)
                    enddo
                                     do i=1,n
              i4(i52+i58+i-1) = i4(2+m+1+i-1)
                                                   enddo
                        if(i4(i52+i75).le.i16)then
             i4(i52+i68) =  i4(i52+i56)
                   endif
    3                                            l(1) = i4(i52+i56)
                              do i = 1,m
                   g(i) = i4(i52+i59+i-1)
                                                  enddo
                                 do i = 1,n
                x(i) = i4(i52+i58+i-1)
                                  enddo
          if(i42.ne.3.and.i42.ne.4.and.i42.ne.5)then
                                      if(i4(i52+i75).le.i16)then
                                        i42 = 1
                                  else
                                              i42 = 2
                                                    endif
                                            endif
                                                   i41 = 1
   86                                                     continue
       if(i42.eq.501.or.i42.eq.601) goto 4
                          i90 = i52+5+m+n
                           do i = 1,m
                                     if( g(i).gt.i2(i)+1.0d-6 )then
                                   i4(i90+i) = 91.0d0
                                           goto 87
                                                            endif
                                       if( g(i).lt.i5(i)-1.0d-6 )then
             i4(i90+i) = 92.0d0
                                              goto 87
                                                    endif
                   if( i5(i).gt.i2(i) )then
                          i4(i90+i) = 93.0d0
                                     goto 87
                        endif
      if( i5(i).eq.i2(i) )then
                                          i4(i90+i) = 90.0d0
               goto 87
                                                 endif
              if( dabs(g(i)-i5(i)) .le. (i2(i)-i5(i))/1000.0d0 )then
                  i4(i90+i) = 0.0d0
                             goto 87
                           endif
      if( dabs(g(i)-i2(i)) .le. (i2(i)-i5(i))/1000.0d0 )then
                                      i4(i90+i) = 22.0d0
            goto 87
                                          endif
                              do j = 1,21
                   if( g(i) .le. i5(i) + j * (i2(i)-i5(i))/21.0d0)then
                                           i4(i90+i) = dble(j)
                                                           goto 87
                            endif
                                                           enddo
   87         continue
            enddo
             goto 4
                             end
                    subroutine i03(m,i8,i4,i32,i49,i19,
     &   i6,i99,o,i18,i1,i36)
                                             implicit none
           integer m,i8,i6,i18,o,i32,i99,i49,i19,i1,i,j,i36(*)
                double precision i4,i33,i320,i76,i80,i79
          dimension i4(i32),i6(i99)
           i76  = dsqrt(dble(i6(i18)))
          i80 = (0.1d0*dble(i36(18))) / i76
       i79 = (1.0d0-1.0d0/dsqrt(dble(i8)+0.1d0)) / dble(i36(10))
                                                 do i=1,m
                      i33 = i4(i19+i-1)
                i320 = i4(i19+i-1)
                      do j=2,i6(o)
                                if(i4(i19+(j-1)*m+i-1).gt.i33)then
                     i33 = i4(i19+(j-1)*m+i-1)
                    endif
                    if(i4(i19+(j-1)*m+i-1).lt.i320)then
        i320 = i4(i19+(j-1)*m+i-1)
                                                  endif
                   enddo
                            i4(i49+i-1) = (i33-i320)/i76
                                 if(i1.gt.i49+i18)then
            i4(i19-(j-1)*m+i-1) = dble(i320)/dble(i33)
                      i33 = i4(i19+i-1)
                 i320 = i4(i19+i-1)
                                                   do j=2,i6(o)
                  if(i4(i19+(j-1)*m+i-1).gt.i33)then
              i33 = i4(i19+(j-1)*m+i-1)
                                endif
       if(i4(i19+(j-1)*m+i-1).lt.i320)then
                                i320 = i4(i19+(j-1)*m+i-1)
             endif
                                     enddo
                      endif
                                  if(i4(i49+i-1).lt.
     &   dabs(i4(i19+i-1))/(dble(10**i36(19))*dble(i6(i18))) )then
                                     i4(i49+i-1)  = dabs(i4(i19+i-1))
     &                             / (dble(10**i36(19))*dble(i6(i18)))
        endif
                                        if(i.gt.m-i8)then
                             if(i4(i49+i-1).lt.i80)then
            i4(i49+i-1) = i80
                    endif
        if(i4(i49+i-1).lt.i79)then
                        i4(i49+i-1) = i79
                                                 endif
                 endif
       enddo
                                                               end
           function i02(s)
                                              implicit none
                                 double precision i02,s,a,b,c,d
              data  a,b,c,d  /0.0d0,0.0d0,0.0d0,0.0d0/
                                if(s.eq.1.2d3)then
                                a =  0.241305836148d+00
              b =  0.502199827430d+00
        c =  0.150532162816d+00
                                          d =  0.278319003121d+00
          endif
                                      if(b.ge.0.5d0)then
                                   s = a + b + c
                                                                   else
                                                   s = a + b + c + d
                 endif
                                             if(s.ge.1.0d0)then
                                                 if(s.ge.2.0d0)then
                                      s = s - 2.0d0
       else
                  s = s - 1.0d0
                                                          endif
            endif
                                                            a = b
                                                        b = c
                                      c = s
                                                                 s = a
         i02 = s
                       end
        subroutine i011(m,i8,o,g,i5,i2,i4,i32,i19,i49,i9,i36)
                                       implicit none
          integer m,i8,t,o,i32,i19,i49,i9,i,j,i36(*)
                     double precision i4,g,i5,i2,i02,i34,i04
                            dimension i4(i32),g(m),i5(m),i2(m)
                                do i=1,m
                                        i34 = i02(i4(1))
                                                     do j=1,o
                               if(i34.le.i4(i9+j-1)) goto 1
                                                            enddo
    1      t = i-1
               g(i) = i4(i19+(j-2)*m+t) + i4(i49+t) *
     &            i04(i34,i02(i4(1)))
                                         if(g(i).lt.i5(i))then
                    if(i34.ge.0.1d0*dble(i36(2)))then
           g(i) = i5(i) + (i5(i)-g(i)) / dble(3**i36(3))
                        if(g(i).gt.i2(i)) g(i) = i2(i)
                   else
                                                          g(i) = i5(i)
                                               endif
                                         goto 2
                                                               endif
                                               if(g(i).gt.i2(i))then
           if(i34.ge.0.1d0*dble(i36(2)))then
                     g(i) = i2(i) - (g(i)-i2(i)) / dble(3**i36(3))
                                  if(g(i).lt.i5(i)) g(i) = i5(i)
                              else
                                        g(i) = i2(i)
                                                       endif
                                       endif
    2                                          continue
                                                      enddo
                                              do i=m-i8+1,m
                                                    g(i) = dnint(g(i))
                   enddo
                                                end
           subroutine i012(i6,i99,i18,i29,i28,i24,i22,i27,i31)
                              implicit none
                         integer i6,i99,i18,i29,i28,i24,i22
                                                      integer i27,i31
                                              dimension i6(i99)
           if(i6(i18).eq.1.and.i6(i22).eq.1)then
                                       i6(i29) = i6(i24)
                                                  else
                     i6(i29) = i6(i28)
                                endif
                   if(i27-i31.gt.i24-i29) i6(1) = 0
          if(i6(i18).le.i6(i22).and.i6(i22).gt.1)then
        i6(i29) = i6(i28) + (i6(i24)-i6(i28)) *
     &               int( dble((i6(i18)-1)) / dble((i6(i22)-1)) )
                                                                 endif
            if(i6(i18).gt.i6(i22).and.i6(i18).lt.2*i6(i22))then
                    i6(i29) = 2 * ( i6(i24) + (i6(i28)-i6(i24)) *
     &      int( dble(i6(i18)) / dble(2*i6(i22)) ) )
                       endif
                              end
                                 subroutine i013(m,i8,g,i5,i2,i4,i32)
                                     implicit none
                                   integer m,i8,i32,i
                                   double precision g,i5,i2,i4,i02
                             dimension g(m),i5(m),i2(m),i4(i32)
                                                       do i=1,m
        g(i) = i5(i) + i02(i4(1)) * (i2(i)-i5(i))
                  if (i.gt.m-i8) g(i) = dnint(g(i))
                         enddo
                                                      end
         subroutine i020(m,o,i4,i32,i19,i14,i40,i11,g,l,i17,p)
                  implicit none
                integer m,o,i32,i19,i14,i40,i11,pface,i,j
          double precision g,l,i17,p,i4
                                        dimension g(m),i4(i32)
                   if(p.ge.i4(i11+o-1))return
                                                            pface = 0
          do i=1,o
         if(p.le.i4(i11+o-i))then
          pface=o-i+1
                   else
                          goto 567
                                             endif
                  enddo
  567                                                do j=1,o-pface
                                                 do i=1,m
                   i4(i19+(o-j)*m+i-1) = i4(i19+(o-j-1)*m+i-1)
                                                                enddo
                                         i4(i14+o-j)     = i4(i14+o-j-1)
              i4(i40+o-j)   = i4(i40+o-j-1)
                                      i4(i11+o-j)     = i4(i11+o-j-1)
               enddo
       do i=1,m
         i4(i19+(pface-1)*m+i-1)  = g(i)
                     enddo
                   i4(i14+pface-1)          = l
                              i4(i40+pface-1)        = i17
                i4(i11+pface-1)          = p
                                                                    end
                  subroutine i08(f,m,i8,n,i0,g,l,x,i5,i2,i16,
     &                   i69,i25,i42,i41,i46,i4,i32,i6,i99,
     &               i30,i68,i48,p,i17,i26,i36)
                                                          implicit none
                       logical i41
      integer i31,i27,i23,i66,i45,i19,i14,i40,i11,i,j,c,i30,
     & i93,i9,w,i49,i1,i7,i170,o,i12,i18,i29,i13,i28,i24,i22,
     &         f,m,i8,n,i0,i69,i25,i42,i46,i32,i6,i99,i26
             double precision g,l,x,i5,i2,i16,i4,i68,i48(*),i02
        dimension g( f*m ),l( f ),x( f*n+1 ),i5(m),i2(m),i4(i32),i6(i99)
                             double precision p(i26),i17(i26)
                                  integer i65, i78, i36(*)
      data i93,i31,i27,i23,i66,i45,i19,i14,i40,i11,i22,
     &  i9,w,i49,i1,i7,i170,o,i12,i18,i29,i13,i28,i24,i78
     &   /0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0/
                                 if(i42.ge.0)then
       call i017(m,n,i31,i27,i23,i66,i45,i30,
     &              i19,i14,i40,i11,w,i49,i9,i1,i7,i170)
         call i016(i12,i29,o,i18,i13,i28,i24,i22)
                                i4(1) = 1.2d3
                 do i=1,i46
                                               i4(1) = i02(i4(1))
                                       enddo
                call i014(i17( 1 ),x( 1 ),n,i0,i16)
                           i4(i27)   = l( 1 )
                    i4(i66) = i17( 1 )
           do i=1,m
                                            i4(i31+i-1) = g( i )
                    enddo
                                                        do i=1,n
                                                i4(i23+i-1) = x( i )
                                                   enddo
            call i05(l( 1 ),i48,i17( 1) ,i16,i42)
                                  if(i42.eq.5) return
                      if(i6(f+2*m+32).ne.74-i6(m+m+f+33))then
                 goto 999
              endif
                                                           goto 101
                   endif
                                                             do c = 1,f
        call i014(i17( c ),x( (c-1)*n+1 ),n,i0,i16)
      if(n.gt.0) call i010(p( c ),l( c ),i17( c ),i4(i45),i16)
           if(n.eq.0) p( c ) = l( c )
             if(i42.gt.-30.or.i42.lt.-40)then
             call i020(m,i6(o),i4,i32,i19,i14,i40,i11,
     &                     g( (c-1)*m+1 ),l( c ),i17( c ),p( c ))
                              endif
             if(i42.le.-30.and.i42.ge.-40)then
           call i019(f,c,m,i4,i32,i6,i99,i19,i14,i40,i11,
     & g( (c-1)*m+1 ),l( c ),i17( c ),p( c ),i36)
                                                  endif
           if(i17( c ).lt.i4(i66)) goto 123
           if(i17( c ).eq.i4(i66).and.l( c ).lt.i4(i27)) goto 123
                                     goto 100
  123     i4(i27)   = l( c )
      i4(i66) = i17( c )
                                           do i=1,m
                                 i4(i31+i-1) = g( (c-1)*m+i )
           enddo
               do i=1,n
                                         i4(i23+i-1) = x( (c-1)*n+i )
                 enddo
      call i05(l( c ),i48,i17( c ),i16,i42)
             if(i42.eq.5) return
  100                      continue
                                       enddo
  101    if(i41)goto 999
                                     if(i42.le.-90)then
          if(i4(i170).gt.i16.and.i4(i40).lt.i4(i170))goto 81
                         if(i4(i170).le.i16.and.i4(i40).le.i16.
     &                         and.i4(i14).lt.i4(i7))goto 81
                              goto 82
   81                               i6(11) = 1
                                           goto 83
   82                  i6(11) = 0
   83            continue
             endif
                                 if(i42.eq.-10)then
            if(i4(i11).lt.i4(i1)-dabs(i4(i1))/dble(2*i36(11))) goto 84
                        i6(13) = 0
                                      goto 85
   84                                            i6(13) = 1
                        i4(i1) = i4(i11)
   85                        continue
                                      endif
              i65 = min(10+m*i36(8),5*i36(20))
                                             if(i6(i18).ge.i65)then
                                                i42 = -95
                                                  endif
              i69 = 0
                           i25 = 0
            if(i6(i18).eq.1.and.i42.eq.-10)then
                                             i78 = 0
                        endif
                               if(i42.eq.-10) i78 = i78 + 1
      if(i31+i31.lt.i27-i31) goto 101
      if(i41) goto 3
                                          if(i42.eq. -1)      goto 13
                                                   if(i42.eq. -2)then
                                            i42 = -1
                           goto 13
                                                   endif
                                         if(i42.eq. -3)then
                            if(i6(i13).ge.i6(i29))then
                                                 i42 = -30
                                                       goto 14
                                                              endif
                     i42 = -1
                                                 goto 13
                                                             endif
                                   if(i42.eq.-30)then
                                                            i42 = -31
                    goto 14
                                         endif
                                           if(i42.le.-31.and.
     &                  i42.ge.-39)then
                               i42 = i42
                                            goto 14
                                   endif
                    if(i42.eq.-40)then
                             i42 = -2
                 goto 12
                                                             endif
                             if(i42.eq.-10)then
                       i42 = -30
                                goto 14
                     endif
      if(i42.le.-90)then
                                             i42 = -3
                                                              goto 11
                            endif
                                       if(i42.eq.  0)then
        i42 = -3
                                                       goto 11
                                          endif
   11         i6(i12) = i6(i12)+1
                       i6(i18) = 0
       call i09(m,i4,i32,i27,i66,i45,i16,
     &          i19,i14,i40,i11,i6,i99,i12,o,i28,i24,i22,
     &                                         i68,i48,i36)
            call i021(i6(o),i4,i32,w)
              if(i23-3*i31.gt.i6(i18)+1)then
                    i4(i9)=0.0d0
                                          do j=1,i6(o)
            i4(i9+j) = i4(i9+j-1) + i4(w+j-1)
                                                        enddo
                                     i4(i7)   = i4(i27)
                       i4(i170) = i4(i66)
                                if(i6(i12).eq.1)then
           if(n.gt.0) call i010(p( 1 ),l( 1 ),i17( 1 ),i4(i45),i16)
                if(n.eq.0) p( 1 ) = l( 1 )
        call i020(m,i6(o),i4,i32,i19,i14,i40,i11,
     &                                   g( 1 ),l( 1 ),i17( 1 ),p( 1 ))
                           endif
               i6(o) = i42
                                  do c = 1,f
       i6(i13) = i6(i13) + 1
                                      if(i6(i18).eq.1)then
               if(i6(10).le.1)then
                                if(i48(6).ne.0.0d0)then
                    call i018(m,i8,g( (c-1)*m+1 ),i5,i2,
     &  i4,i32,i31,dabs(i48(6)),i36)
                                                              else
                call i013(m,i8,g( (c-1)*m+1 ),i5,i2,i4,i32)
              endif
                                    else
        call i06(m,i8,i4,i32,i6,i99,i31,
     &                 g( (c-1)*m+1 ),i5,i2,dabs(i48(6)),i36)
                  endif
                                             endif
         if(i6(i18).gt.1)call i011(m,i8,i6(o),g( (c-1)*m+1 ),i5,i2,
     &   i4,i32,i19,i49,i9,i36)
        if(i18.lt.i1-i49) call i013(m,i8,g( (c-1)*m+1 ),i5,i2,i4,i32)
                                     enddo
                    endif
                            i4(i9)=0.0d0
             do j=1,i6(o)
                               i4(i9+j) = i4(i9+j-1) + i4(w+j-1)
                         enddo
                                                  i4(i7)   = i4(i27)
       i4(i170) = i4(i66)
                                                           i93 = 0
                                                 if(i6(i12).eq.1)then
           if(n.gt.0) call i010(p( 1 ),l( 1 ),i17( 1 ),i4(i45),i16)
                 if(n.eq.0) p( 1 ) = l( 1 )
                 call i020(m,i6(o),i4,i32,i19,i14,i40,i11,
     &                   g( 1 ),l( 1 ),i17( 1 ),p( 1 ))
             endif
   12                                            i6(i18) = i6(i18) + 1
                                       i6(i13) = 0
         call i03(m,i8,i4,i32,i49,i19,i6,i99,o,i18,i1,i36)
                                 if(i48(7).gt.0.0d0)then
                                    i6(i29) = i6(i28)
                               else
           call i012(i6,i99,i18,i29,i28,i24,i22,i27,i31)
                                 endif
              if(i6(i18).eq.1) i4(i1) = 1.0d16
         if(i6(i18).gt.1) i4(i1) = i4(i11)
   13                                             do c = 1,f
           i6(i13) = i6(i13) + 1
                                       if(i6(i18).eq.1)then
                                   if(i6(10).le.1)then
             if(i48(6).ne.0.0d0)then
            call i018(m,i8,g( (c-1)*m+1 ),i5,i2,
     & i4,i32,i31,dabs(i48(6)),i36)
                                else
                     call i013(m,i8,g( (c-1)*m+1 ),i5,i2,i4,i32)
            endif
                                                              else
                          call i06(m,i8,i4,i32,i6,i99,i31,
     &          g( (c-1)*m+1 ),i5,i2,dabs(i48(6)),i36)
                                                 endif
                                                      endif
          if(i6(i18).gt.1)call i011(m,i8,i6(o),g( (c-1)*m+1 ),i5,i2,
     &                                   i4,i32,i19,i49,i9,i36)
        if(i18.lt.i1-i49) call i013(m,i8,g( (c-1)*m+1 ),i5,i2,i4,i32)
                 enddo
                 if(i6(i13).ge.i6(i29).and.i42.ne.-3) i42 = -10
    3                                                       return
   14                                continue
                                 if(i6(13).eq.1.or.i6(i18).eq.1)then
       i42 = -2
               goto 12
          else
             if(i42.lt.-30.and.i6(31).eq.1)then
                                i42 = -2
                                                   goto 12
                                  endif
                        if(i42.eq.-39)then
                                       i93 = 1
                             i42    = -99
               goto 101
                   endif
                                                    do c = 1,f
               if(f.gt.1) i6(31) = 0
                    call i015(f,m,i8,g( (c-1)*m+1 ),i5,i2,i19,i49,i4,
     &                                  i32,i6,i99,i42,i93,i36)
                            if(i42.eq.-30.and.f.gt.1) i42 = -31
                     if((i14-i19)/i30-i18.gt.0)then
                                                          i42 = -23
                  goto 14
                        endif
                     if(i93.eq.1.and.c.gt.1)then
                                call i07(m,i8,g( (c-1)*m+1 ),i5,i2,
     &                          i4,i32,i6,i99,i18,i19,i36)
                                                i93 = 0
            i42 = -39
                       endif
                 enddo
                             if(i93.eq.1) goto 101
        goto 3
                                     endif
  999      l( 1 )   = i4(i27)
                      i17( 1 ) = i4(i66)
                                                               do i=1,m
                           g(i) = i4(i31+i-1)
                                     enddo
                                                             do j=1,n
                                    x(j) = i4(i23+j-1)
                                                        enddo
           if(i17( 1 ).le.i16)then
                       i42 = 0
                    else
                                          i42 = 1
               endif
                                                   end
        subroutine i06(m,i8,i4,i32,i6,i99,i31,g,i5,i2,i47,i36)
          implicit none
              integer m,i8,i32,i31,i6,i99,i,i36(*)
                double precision i4,g,i5,i2,i47,i35,i02,i04
               double precision i34
       dimension i4(i32),i6(i99),g(m),i5(m),i2(m)
          do i=1,m
               i35 = (i2(i)-i5(i)) / dble(i6(10))
             if(i.gt.m-i8.and.i35.lt.0.1d0*dble(i36(13)))then
                                     i35 = 0.1d0*dble(i36(13))
                                     endif
                          if(i47.gt.0.0d0)then
                     if(i35.gt.(i2(i)-i5(i))/i47)then
                        i35 = (i2(i)-i5(i))/i47
                             endif
                     if(i.gt.m-i8)then
          if(i35.lt.1.0d0/dsqrt(i47))then
                                      i35 = 1.0d0 / dsqrt(i47)
                     endif
                                                              endif
                                                                  endif
                                                 i34 = i02(i4(1))
                         g(i) = i4(i31+i-1) + i35 *
     &  i04(i34,i02(i4(1)))
                         if(g(i).lt.i5(i))then
                                    if(i34.ge.0.1d0*dble(i36(2)))then
               g(i) = i5(i) + (i5(i)-g(i)) / dble(3**i36(3))
                            if(g(i).gt.i2(i)) g(i) = i2(i)
          else
                                                    g(i) = i5(i)
                                     endif
               goto 2
                     endif
                                               if(g(i).gt.i2(i))then
               if(i34.ge.0.1d0*dble(i36(2)))then
         g(i) = i2(i) - (g(i)-i2(i)) / dble(3**i36(3))
                   if(g(i).lt.i5(i)) g(i) = i5(i)
                                                        else
                        g(i) = i2(i)
                    endif
                             endif
    2                                 if(i.gt.m-i8) g(i) = dnint(g(i))
                                        enddo
                                           end
           subroutine i019(f,c,m,i4,i32,i6,i99,i19,i14,i40,i11,
     &                                  g,l,i17,p,i36)
                                                implicit none
                    integer f,c,m,i32,i99,i6,i19,i14,i40,i11,i,i36(*)
             double precision g,l,i17,p,i4
                               dimension g(m),i4(i32),i6(i99)
                          if(i17.le.0.0d0.and.i4(i40).le.0.0d0)then
         if(l.ge. i4(i14) - dabs(i4(i14)) / dble(10**i36(12)) )then
                                               i6(31 + c) = 0
                goto 1
                endif
             else
         if(p.ge. i4(i11) - dabs(i4(i11)) / dble(10**i36(12)) )then
                                          i6(31 + c) = 0
                   goto 1
         endif
                                                                endif
                   do i = 1,m
                                                    i4(i19+i-1) = g(i)
              enddo
        i4(i40) = i17
             i4(i14)   = l
                      i4(i11)   = p
                              i6(31 + c) = 1
   1                                                if(c.eq.f)then
                                           i6(31) = 0
                                               do i = 1,f
                                              i6(31) = i6(31) + i6(31+i)
             enddo
                            if(i6(31).gt.1) i6(31) = 1
                                               endif
         return
      end
                 subroutine i016(i12,i29,o,i18,i13,i28,i24,i22)
                                                implicit none
             integer i12,i29,o,i18,i13,i28,i24,i22
                                   o      = 1
                                      i12    = 2
          i29   = 3
                                                        i18    = 4
                                    i13    = 5
                  i28   = 6
                          i24   = 7
            i22   = 8
                                                  end
        subroutine i017(m,n,i31,i27,i23,i66,i45,i30,
     &                  i19,i14,i40,i11,w,i49,i9,i1,i7,i170)
                                   implicit none
             integer m,n
         integer i31,i27,i23,i66
                                               integer i45
                                                            integer i30
                              integer i19,i14,i40,i11
          integer w,i49
                           integer i9
        integer i1,i7,i170
                                i31        = 2
                                        i27        = i31        + m
             i23        = i27        + 1
                   i66      = i23        + n
                       i45       = i66      + 1
       i19         = i45       + 1
                               i14         = i19         + m * i30
                 i40       = i14         + i30
                                     i11         = i40       + i30
         i9          = i11         + i30
       w           = i9          + i30 + 1
       i49       = w           + i30
           i1          = i49       + m
             i7          = i1          + 1
                   i170        = i7          + 1
                                                end
        subroutine i018(m,i8,g,i5,i2,i4,i32,i31,i47,i36)
                                                        implicit none
                                  integer m,i8,i32,i31,i,i36(*)
           double precision g,i5,i2,i4,i47,i35,i02,i04,i34
           dimension g(m),i5(m),i2(m),i4(i32)
                            do i=1,m
              i35 = (i2(i)-i5(i)) / i47
                                          if(i.gt.m-i8)then
                if(i35.lt.1.0d0/dsqrt(i47))then
                                             i35 = 1.0d0 / dsqrt(i47)
          endif
             endif
              i34 = i02(i4(1))
                   g(i) = i4(i31+i-1) + i35 *
     &                        i04(i34,i02(i4(1)))
            if(g(i).lt.i5(i))then
                                if(i34.ge.0.1d0*dble(i36(2)))then
                  g(i) = i5(i) + (i5(i)-g(i)) / dble(3**i36(3))
                if(g(i).gt.i2(i)) g(i) = i2(i)
                                 else
                       g(i) = i5(i)
        endif
                                                      goto 2
               endif
                                      if(g(i).gt.i2(i))then
                          if(i34.ge.0.1d0*dble(i36(2)))then
           g(i) = i2(i) - (g(i)-i2(i)) / dble(3**i36(3))
                 if(g(i).lt.i5(i)) g(i) = i5(i)
                 else
                                            g(i) = i2(i)
                  endif
                                                       endif
    2                         if(i.gt.m-i8) g(i) = dnint(g(i))
                                 enddo
                                                  end
                                 subroutine i010(p,l,i17,i45,i16)
                                            implicit none
                              double precision p,l,i17,i45,i16,i61
                  i61 = l - i45
                           if (l.le.i45.and.i17.le.i16) then
                                                        p = i61
                                       return
                                                          else
       if (l.le.i45) then
             p = i17
                                                                 return
             else
                                              if (i17.le.i61) then
      p = i61 + i17**2/(2.0d0*i61) - i17/2.0d0
                                                       else
                     p = i17 + i61**2/(2.0d0*i17) - i61/2.0d0
                       endif
                                      endif
                                        endif
           end
                                       subroutine i021(o,i4,i32,w)
                                 implicit none
              integer o,i32,j,i57,w
                    double precision i4
                                                       dimension i4(i32)
                                 i57 = 0
                                   do j=1,o
                                                      i57 = i57 + j
                                                                enddo
               do j=1,o
                               i4(w+j-1) = dble(o-j+1)/dble(i57)
                                                              enddo
                                           end
       subroutine i014(i17,x,n,i0,i16)
                    implicit none
                                                    integer n,i0,i
             double precision x,i17,i16
         dimension x(n)
            i17 = 0.0d0
                       if(n.eq.0)return
                                    do i=1,n
                                     if (x(i).lt.-i16) i17 = i17 - x(i)
                                        enddo
                                                  do i=1,i0
       if (x(i).gt. i16) i17 = i17 + x(i)
          enddo
                                                           end
                                       subroutine i024(i36,m,i8)
                                    implicit none
         integer i36(*),m,i8
                   if(m-i8.gt.0)then
                                     i36(   1) =                 18
              i36(   2) =                  6
          i36(   3) =                  4
               i36(   4) =                  4
                            i36(   5) =                  5
      i36(   6) =                 18
                           i36(   7) =                 13
              i36(   8) =                  5
                         i36(   9) =                  2
                  i36(  10) =                  7
                                   i36(  11) =                 15
                              i36(  12) =                  8
                          i36(  13) =                  9
                                  i36(  14) =                  7
                                  i36(  15) =                 12
                                     i36(  16) =                  5
             i36(  17) =                  5
                           i36(  18) =                  5
                              i36(  19) =                  8
             i36(  20) =                 37
                 else
       i36(   1) =                  4
                    i36(   2) =                  4
                           i36(   3) =                  3
       i36(   4) =                  3
                                 i36(   5) =                  2
                    i36(   6) =                 49
                                       i36(   7) =                 37
          i36(   8) =                   7
            i36(   9) =                  0
              i36(  10) =                  8
                        i36(  11) =                  3
                                      i36(  12) =                  6
                   i36(  13) =                  4
                         i36(  14) =                  3
                                    i36(  15) =                  3
                        i36(  16) =                  2
                            i36(  17) =                 10
                              i36(  18) =                 12
                         i36(  19) =                  4
                          i36(  20) =                 58
                                     endif
                           end
      subroutine alphabet(a,b)
      implicit none
      character*1 a
      integer b
      b = 0
      if(a(1:1).eq.'A') b = 52
      if(a(1:1).eq.'B') b = 28
      if(a(1:1).eq.'C') b = 49
      if(a(1:1).eq.'D') b = 30
      if(a(1:1).eq.'E') b = 31
      if(a(1:1).eq.'F') b = 32
      if(a(1:1).eq.'G') b = 33
      if(a(1:1).eq.'H') b = 34
      if(a(1:1).eq.'I') b = 35
      if(a(1:1).eq.'J') b = 36
      if(a(1:1).eq.'K') b = 37
      if(a(1:1).eq.'L') b = 38
      if(a(1:1).eq.'M') b = 39
      if(a(1:1).eq.'N') b = 40
      if(a(1:1).eq.'O') b = 41
      if(a(1:1).eq.'P') b = 42
      if(a(1:1).eq.'Q') b = 43
      if(a(1:1).eq.'R') b = 44
      if(a(1:1).eq.'S') b = 45
      if(a(1:1).eq.'T') b = 46
      if(a(1:1).eq.'U') b = 47
      if(a(1:1).eq.'V') b = 48
      if(a(1:1).eq.'W') b = 29
      if(a(1:1).eq.'X') b = 50
      if(a(1:1).eq.'Y') b = 51
      if(a(1:1).eq.'Z') b = 27
      if(a(1:1).eq.'0') b = 53
      if(a(1:1).eq.'1') b = 54
      if(a(1:1).eq.'2') b = 55
      if(a(1:1).eq.'3') b = 56
      if(a(1:1).eq.'4') b = 57
      if(a(1:1).eq.'5') b = 58
      if(a(1:1).eq.'6') b = 59
      if(a(1:1).eq.'7') b = 60
      if(a(1:1).eq.'8') b = 61
      if(a(1:1).eq.'9') b = 62
      if(a(1:1).eq.'a') b = 23
      if(a(1:1).eq.'b') b = 2
      if(a(1:1).eq.'c') b = 3
      if(a(1:1).eq.'d') b = 16
      if(a(1:1).eq.'e') b = 5
      if(a(1:1).eq.'f') b = 13
      if(a(1:1).eq.'g') b = 7
      if(a(1:1).eq.'h') b = 8
      if(a(1:1).eq.'i') b = 9
      if(a(1:1).eq.'j') b = 10
      if(a(1:1).eq.'k') b = 11
      if(a(1:1).eq.'l') b = 12
      if(a(1:1).eq.'m') b = 6
      if(a(1:1).eq.'n') b = 14
      if(a(1:1).eq.'o') b = 15
      if(a(1:1).eq.'p') b = 4
      if(a(1:1).eq.'q') b = 17
      if(a(1:1).eq.'r') b = 18
      if(a(1:1).eq.'s') b = 19
      if(a(1:1).eq.'t') b = 20
      if(a(1:1).eq.'u') b = 21
      if(a(1:1).eq.'v') b = 22
      if(a(1:1).eq.'w') b = 1
      if(a(1:1).eq.'x') b = 24
      if(a(1:1).eq.'y') b = 25
      if(a(1:1).eq.'z') b = 26
      if(a(1:1).eq.'_') b = 64
      if(a(1:1).eq.'(') b = 65
      if(a(1:1).eq.')') b = 66
      if(a(1:1).eq.'+') b = 67
      if(a(1:1).eq.'-') b = 68
      if(a(1:1).eq.'&') b = 69
      if(a(1:1).eq.'.') b = 70
      if(a(1:1).eq.',') b = 71
      if(a(1:1).eq.':') b = 72
      if(a(1:1).eq.';') b = 73
      if(a(1:1).eq.'*') b = 74
      if(a(1:1).eq.'=') b = 75
      if(a(1:1).eq.'/') b = 76
      if(a(1:1).eq.'!') b = 80
      if(a(1:1).eq.'[') b = 83
      if(a(1:1).eq.']') b = 84
      end



CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     This subroutine handles all printing commands for MIDACO.
C     Note that this subroutine is called independently from MIDACO and
C     MIDACO itself does not include any print commands (due to
C     compiler portability and robustness).
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE MIDACO_PRINT(C,PRINTEVAL,SAVE2FILE,IFLAG,ISTOP,F,G,
     &           X,XL,XU,N,NI,M,ME,RW,LRW,MAXEVAL,MAXTIME,PARAM,P,KEY)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT NONE
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      INTEGER C,PRINTEVAL,SAVE2FILE,IFLAG,ISTOP,EVAL,N,NI,M,ME,LRW
      INTEGER TIC,Q,I,IOUT1,IOUT2,MAXEVAL,MAXTIME,KMAX,UPDATE
      INTEGER KF,KG,KRES,KX,WF,WG,WRES,WX,KBEST,WBEST,P
      DOUBLE PRECISION TNOW,TSTART,TMAX,RW(LRW),BESTF,BESTR,F,G(M),X(N)
      DOUBLE PRECISION XL(N),XU(N),ACC,PARAM(9),DUMMY_F,DUMMY_VIO
C     Increase size, if problems with N > 1000 are solved
      DOUBLE PRECISION BESTX(1000),BESTG(1000)
      CHARACTER*60 KEY
      DATA KF,KG,KRES,KX,WF,WG,WRES,WX /0,0,0,0,0,0,0,0/
      DATA KBEST,WBEST,TIC,Q,EVAL,IOUT1,IOUT2 /0,0,0,0,0,0,0/
      DATA BESTF,BESTR,DUMMY_F,DUMMY_VIO /0.0D0,0.0D0,0.0D0,0.0D0/
      DATA TNOW,TSTART,TMAX,ACC /0.0D0,0.0D0,0.0D0,0.0D0/
      IF(C.EQ.2)THEN
          CALL GET_TIME(TNOW)
          TNOW = TNOW - TSTART
          EVAL = EVAL + P
          IF(IFLAG.GE.10)THEN
              CALL WARNINGS_AND_ERRORS( IFLAG, 0 )
              IF(SAVE2FILE.GE.1)THEN
                  CALL WARNINGS_AND_ERRORS( IFLAG, IOUT1 )
                  CALL WARNINGS_AND_ERRORS( IFLAG, IOUT2 )
              ENDIF
              IF(IFLAG.GE.100) RETURN
          ENDIF
          IF(PRINTEVAL.GE.1)THEN
              TIC = TIC + P
              IF(TIC.GE.PRINTEVAL.OR.EVAL.EQ.P.OR.IFLAG.GE.1)THEN
                  IF(EVAL.GT.P) TIC = 0
                  IF(RW(KRES).EQ.RW(WRES))THEN
                    KBEST=KF
                    WBEST=WF
                  ELSE
                    KBEST=KRES
                    WBEST=WRES
                  ENDIF
                  IF(RW(WBEST).LT.RW(KBEST).OR.
     &              (IFLAG.GE.1.OR.IFLAG.EQ.-300))THEN
                      BESTF = RW(WF)
                      BESTR = RW(WRES)
                      DO I=1,M
                        BESTG(I) = RW(WG+I)
                      ENDDO
                      DO I=1,N
                        BESTX(I) = RW(WX+I)
                      ENDDO
                  ELSE
                      BESTF = RW(KF)
                      BESTR = RW(KRES)
                      DO I=1,M
                        BESTG(I) = RW(KG+I)
                      ENDDO
                      DO I=1,N
                        BESTX(I) = RW(KX+I)
                      ENDDO
                  ENDIF
                  CALL PRINT_LINE(EVAL,TNOW,BESTF,BESTR,0)
                  IF(SAVE2FILE.GE.1)THEN
                    CALL PRINT_LINE(EVAL,TNOW,BESTF,BESTR,IOUT1)
                  ENDIF
                  IF(SAVE2FILE.GE.1)THEN
                    UPDATE = 0
                    IF( (BESTR.LT.DUMMY_VIO) .OR.
     &                  (BESTR.EQ.DUMMY_VIO.AND.BESTF.LT.DUMMY_F) )THEN
                       DUMMY_F   = BESTF
                       DUMMY_VIO = BESTR
                       UPDATE    = 1
                    ENDIF

                    IF(UPDATE.EQ.1)THEN
                     WRITE(IOUT2,31)
   31                FORMAT(/,/,'            CURRENT BEST SOLUTION')
                     CALL PRINT_SOLUTION( N, M, ME, BESTX, BESTG, BESTF,
     &                     BESTR, XL, XU, ACC, EVAL, TNOW, IFLAG, IOUT2)
                     CONTINUE
                    ENDIF
                    CALL FORCE_OUTPUT( IOUT1 )
                    CALL FORCE_OUTPUT( IOUT2 )
                  ENDIF
              ENDIF
          ENDIF
          IF(ISTOP.EQ.0)THEN
              IF(TNOW.GE.TMAX     ) IFLAG = -999
              IF(EVAL.GE.MAXEVAL-1)THEN
                IF(MAXEVAL.LE.99999999) IFLAG = -999
              ENDIF
          ENDIF
          RETURN
      ENDIF
      IF(C.EQ.1)THEN
          IFLAG = 0
          ISTOP = 0
          TMAX = DBLE(MAXTIME)
          CALL GET_TIME(TSTART)
          EVAL = 0
          if(param(1).le.0.0D0)then
              acc = 1.0D-3
          else
              acc = param(1)
          endif
          KMAX = 100
          Q    = 2*N+M+(N+5)*KMAX+8
          KX   = 1
          KF   = 2+N
          KG   = 2+N
          KRES = 2+N+1+M
          WX   = Q
          WF   = Q+1+N
          WG   = Q+1+N
          WRES = Q+1+N+1+M
          IF(SAVE2FILE.GE.1)THEN
              IOUT1 = 36
              IOUT2 = 37
              OPEN(IOUT1,FILE='MIDACO_SCREEN.TXT'  ,STATUS='UNKNOWN')
              OPEN(IOUT2,FILE='MIDACO_SOLUTION.TXT',STATUS='UNKNOWN')
          ENDIF
          BESTF = 1.0D+32
          BESTR = 1.0D+32
          DUMMY_F   = 1.0D+32
          DUMMY_VIO = 1.0D+32
          TIC = 0
          IF(PRINTEVAL.GE.1)THEN
              CALL PRINT_HEAD( N, NI, M, ME, PARAM, MAXEVAL, MAXTIME,
     &                         PRINTEVAL, SAVE2FILE, KEY, 0)
              IF(SAVE2FILE.GE.1)THEN
                CALL PRINT_HEAD( N, NI, M, ME, PARAM, MAXEVAL, MAXTIME,
     &                        PRINTEVAL, SAVE2FILE, KEY, IOUT1)
              ENDIF
          ENDIF
          IF(SAVE2FILE.GE.1) WRITE(IOUT2,3)
    3     FORMAT('MIDACO - SOLUTION',/,
     &           '-----------------',/,
     &'This file saves the current best solution X found by MIDACO.',/,
     &'This file is updated after every PRINTEVAL function evaluation,',
     &/,'if X has been improved.',/,/)
          CALL FORCE_OUTPUT( IOUT1 )
          CALL FORCE_OUTPUT( IOUT2 )
      ENDIF
      IF(C.EQ.3.AND.PRINTEVAL.GE.1)THEN
          CALL PRINT_FINAL( IFLAG,TNOW,TMAX,EVAL,MAXEVAL,
     &                      N,M,ME,X,G,F,XL,XU,RW,ACC,WRES,PARAM,
     &                      0 )
          IF(SAVE2FILE.GT.0)THEN
             CALL PRINT_FINAL( IFLAG,TNOW,TMAX,EVAL,MAXEVAL,
     &                         N,M,ME,X,G,F,XL,XU,RW,ACC,WRES,PARAM,
     &                         IOUT1 )
             CALL PRINT_FINAL( IFLAG,TNOW,TMAX,EVAL,MAXEVAL,
     &                         N,M,ME,X,G,F,XL,XU,RW,ACC,WRES,PARAM,
     &                         IOUT2 )
          ENDIF
          CALL FORCE_OUTPUT( IOUT1 )
          CALL FORCE_OUTPUT( IOUT2 )
      ENDIF
      END
      SUBROUTINE PRINT_HEAD( N, NI, M, ME, PARAM, MAXEVAL, MAXTIME,
     &                       PRINTEVAL, SAVE2FILE, KEY, IOUT)
      IMPLICIT NONE
      INTEGER N,NI,M,ME,IOUT,MAXTIME,MAXEVAL,SAVE2FILE,PRINTEVAL,I
      DOUBLE PRECISION PARAM(*)
      CHARACTER*60 KEY
      WRITE(IOUT,1) KEY,N,MAXEVAL,NI,
     &              MAXTIME,M,PRINTEVAL,ME,SAVE2FILE
      DO I = 1,9
          IF(PARAM(I).NE.0.0D0) GOTO 66
      ENDDO
      IF(PRINTEVAL.GE.1) WRITE(IOUT,101)
      GOTO 67
   66  IF(PARAM(1).NE.0.0D0.AND.DABS(PARAM(1)).LT.0.000001D0)THEN
          WRITE(IOUT,202) PARAM(1)
       ELSE
          IF(PARAM(1).EQ.0.0D0) WRITE(IOUT,203) 0.001D0
          IF(PARAM(1).NE.0.0D0) WRITE(IOUT,203) PARAM(1)
       ENDIF
       WRITE(IOUT,104) PARAM(2)
       IF(PARAM(3).NE.0.0D0)THEN
          WRITE(IOUT,205) PARAM(3)
       ELSE
          WRITE(IOUT,204) PARAM(3)
       ENDIF
       WRITE(IOUT,105) PARAM(4)
       IF(PARAM(5).NE.0.0D0)THEN
          WRITE(IOUT,207) PARAM(5)
       ELSE
          WRITE(IOUT,206) PARAM(5)
       ENDIF
       WRITE(IOUT,106) PARAM(6),PARAM(7),PARAM(8),PARAM(9)
   67 CONTINUE
       WRITE(IOUT,103)
    1 FORMAT(/,
     &       ' MIDACO 4.0    (www.midaco-solver.com)',/,
     &       ' -------------------------------------',/,/,
     &       ' LICENSE-KEY:  ',A60,/,/,
     &       ' -------------------------------------',/,
     &       ' | N', I7, '    | MAXEVAL',    I12,' |',/,
     &       ' | NI',I6, '    | MAXTIME',    I12,' |',/,
     &       ' | M', I7, '    | PRINTEVAL',  I10,' |',/,
     &       ' | ME',I6 ,'    | SAVE2FILE',  I10,' |',/,
     &       ' |-----------------------------------|')
  101 FORMAT(' | PARAMETER:   All by default (0)   |',/,
     &       ' -------------------------------------')
  202 FORMAT(' | PARAM(1)', D10.1,'  ACCURACY G(X) |')
  203 FORMAT(' | PARAM(1)', F10.6,'  ACCURACY G(X) |')
  104 FORMAT(' |-----------------------------------|',/,
     &       ' | PARAM(2)  ', F10.1,'  RANDOM-SEED |')
  204 FORMAT(' | PARAM(3)  ', F10.1,'  FSTOP       |')
  205 FORMAT(' | PARAM(3)  ', D10.3,'  FSTOP       |')
  105 FORMAT(' | PARAM(4)  ', F10.1,'  AUTOSTOP    |')
  206 FORMAT(' | PARAM(5)  ', F10.1,'  ORACLE      |')
  207 FORMAT(' | PARAM(5)  ', D10.3,'  ORACLE      |')
  106 FORMAT(' | PARAM(6)  ', F10.1,'  FOCUS       |',/,
     &       ' | PARAM(7)  ', F10.1,'  ANTS        |',/,
     &       ' | PARAM(8)  ', F10.1,'  KERNEL      |',/,
     &       ' | PARAM(9)  ', F10.1,'  CHARACTER   |',/,
     &       ' -------------------------------------')
  103   FORMAT(/,/,
     &  ' [     EVAL,    TIME]        OBJECTIVE FUNCTION VALUE    ',
     &  '     VIOLATION OF G(X)',/,
     &  ' -----------------------------------------------',
     &  '-------------------------------')
      END
      SUBROUTINE PRINT_LINE( EVAL, TNOW, F, VIO, IOUT)
      IMPLICIT NONE
      INTEGER EVAL, IOUT
      DOUBLE PRECISION TNOW, F, VIO
      IF(DABS(F).LE.1.0D+10)THEN
        IF(VIO.LE.1.0D+5)THEN
           WRITE(IOUT,1) EVAL, TNOW, F, VIO
        ELSE
           WRITE(IOUT,2) EVAL, TNOW, F, VIO
        ENDIF
      ELSE
        IF(VIO.LE.1.0D+5)THEN
           WRITE(IOUT,3) EVAL, TNOW, F, VIO
        ELSE
           WRITE(IOUT,4) EVAL, TNOW, F, VIO
        ENDIF
      ENDIF
    1 FORMAT(
     &' [',I9,',',F8.2,']        F(X):',F19.8,'         VIO:',F13.6)
    2 FORMAT(
     &' [',I9,',',F8.2,']        F(X):',F19.8,'         VIO:',D13.6)
    3 FORMAT(
     &' [',I9,',',F8.2,']        F(X):',D19.10,'         VIO:',F13.6)
    4 FORMAT(
     &' [',I9,',',F8.2,']        F(X):',D19.10,'         VIO:',D13.6)
      END
      SUBROUTINE PRINT_SOLUTION( N, M, ME, X, G, F, VIO,
     &                           XL, XU, ACC, EVAL, TNOW, IFLAG, IOUT)
      IMPLICIT NONE
      INTEGER N,M,ME,EVAL,IFLAG,IOUT,I,J
      DOUBLE PRECISION X(*),G(*),F,VIO,XL(*),XU(*),ACC,TNOW,PROFIL
      WRITE(IOUT,4)
      WRITE(IOUT,41) EVAL,TNOW,IFLAG
      IF(DABS(F).LE.1.0D+14)THEN
          WRITE(IOUT,42) F
      ELSE
          WRITE(IOUT,82) F
      ENDIF
          IF(M.GT.0) WRITE(IOUT,142)
          IF(IFLAG.LT.100)THEN
              IF(VIO.LE.1.0D+12)THEN
                WRITE(IOUT,183) VIO
              ELSE
                WRITE(IOUT,184) VIO
              ENDIF
          ENDIF
          IF(M.GT.0) WRITE(IOUT,144)
          DO I = 1,ME
              IF(DABS(G(I)).LE.ACC)THEN
                  IF(DABS(G(I)).LE.1.0D+14)THEN
                      WRITE(IOUT,43) I,G(I)
                  ELSE
                      WRITE(IOUT,83) I,G(I)
                  ENDIF
              ELSE
                  IF(DABS(G(I)).LE.1.0D+14)THEN
                      WRITE(IOUT,431) I,G(I)
                  ELSE
                      WRITE(IOUT,831) I,G(I)
                  ENDIF
              ENDIF
          ENDDO
          DO I = ME+1,M
              IF(G(I).GE.-ACC)THEN
                  IF(DABS(G(I)).LE.1.0D+14)THEN
                      WRITE(IOUT,44) I,G(I)
                  ELSE
                      WRITE(IOUT,84) I,G(I)
                  ENDIF
              ELSE
                  IF(DABS(G(I)).LE.1.0D+14)THEN
                      WRITE(IOUT,441) I,G(I)
                  ELSE
                      WRITE(IOUT,841) I,G(I)
                  ENDIF
              ENDIF
          ENDDO
          WRITE(IOUT,145)
          DO I = 1,N

          PROFIL = -1.0D0

          IF( X(I).GT.XU(I)+1.0D-6 )THEN
              PROFIL = 91.0D0
              GOTO 88
          ENDIF
          IF( X(I).LT.XL(I)-1.0D-6 )THEN
              PROFIL = 92.0D0
              GOTO 88
          ENDIF
          IF( XL(I).GT.XU(I) )THEN
              PROFIL = 93.0D0
              GOTO 88
          ENDIF
          IF( XL(I).EQ.XU(I) )THEN
              PROFIL = 90.0D0
              GOTO 88
          ENDIF
          IF( DABS(X(I)-XL(I)) .LE. (XU(I)-XL(I))/1000.0D0 )THEN
              PROFIL = 0.0D0
              GOTO 88
          ENDIF
          IF( DABS(XU(I)-X(I)) .LE. (XU(I)-XL(I))/1000.0D0 )THEN
              PROFIL = 22.0D0
              GOTO 88
          ENDIF
          DO J = 1,21
              IF( X(I) .LE. XL(I) + DBLE(J) * (XU(I)-XL(I))/21.0D0)THEN
                  PROFIL = DBLE(J)
                  GOTO 88
              ENDIF
          ENDDO
   88    CONTINUE
         IF(DABS(X(I)).LE.1.0D+14)THEN
             IF(PROFIL.EQ. 0.0D0) WRITE(IOUT,400) I,X(I)
             IF(PROFIL.EQ. 1.0D0) WRITE(IOUT,401) I,X(I)
             IF(PROFIL.EQ. 2.0D0) WRITE(IOUT,402) I,X(I)
             IF(PROFIL.EQ. 3.0D0) WRITE(IOUT,403) I,X(I)
             IF(PROFIL.EQ. 4.0D0) WRITE(IOUT,404) I,X(I)
             IF(PROFIL.EQ. 5.0D0) WRITE(IOUT,405) I,X(I)
             IF(PROFIL.EQ. 6.0D0) WRITE(IOUT,406) I,X(I)
             IF(PROFIL.EQ. 7.0D0) WRITE(IOUT,407) I,X(I)
             IF(PROFIL.EQ. 8.0D0) WRITE(IOUT,408) I,X(I)
             IF(PROFIL.EQ. 9.0D0) WRITE(IOUT,409) I,X(I)
             IF(PROFIL.EQ.10.0D0) WRITE(IOUT,410) I,X(I)
             IF(PROFIL.EQ.11.0D0) WRITE(IOUT,411) I,X(I)
             IF(PROFIL.EQ.12.0D0) WRITE(IOUT,412) I,X(I)
             IF(PROFIL.EQ.13.0D0) WRITE(IOUT,413) I,X(I)
             IF(PROFIL.EQ.14.0D0) WRITE(IOUT,414) I,X(I)
             IF(PROFIL.EQ.15.0D0) WRITE(IOUT,415) I,X(I)
             IF(PROFIL.EQ.16.0D0) WRITE(IOUT,416) I,X(I)
             IF(PROFIL.EQ.17.0D0) WRITE(IOUT,417) I,X(I)
             IF(PROFIL.EQ.18.0D0) WRITE(IOUT,418) I,X(I)
             IF(PROFIL.EQ.19.0D0) WRITE(IOUT,419) I,X(I)
             IF(PROFIL.EQ.20.0D0) WRITE(IOUT,420) I,X(I)
             IF(PROFIL.EQ.21.0D0) WRITE(IOUT,421) I,X(I)
             IF(PROFIL.EQ.22.0D0) WRITE(IOUT,422) I,X(I)
             IF(PROFIL.EQ.90.0D0) WRITE(IOUT,490) I,X(I)
             IF(PROFIL.EQ.91.0D0) WRITE(IOUT,491) I,X(I)
             IF(PROFIL.EQ.92.0D0) WRITE(IOUT,492) I,X(I)
             IF(PROFIL.EQ.93.0D0) WRITE(IOUT,493) I,X(I)
             IF(PROFIL.LT.0.0D0) WRITE(IOUT,*) 'PROFIL-ERROR'
         ELSE
             IF(PROFIL.EQ. 0.0D0) WRITE(IOUT,800) I,X(I)
             IF(PROFIL.EQ. 1.0D0) WRITE(IOUT,801) I,X(I)
             IF(PROFIL.EQ. 2.0D0) WRITE(IOUT,802) I,X(I)
             IF(PROFIL.EQ. 3.0D0) WRITE(IOUT,803) I,X(I)
             IF(PROFIL.EQ. 4.0D0) WRITE(IOUT,804) I,X(I)
             IF(PROFIL.EQ. 5.0D0) WRITE(IOUT,805) I,X(I)
             IF(PROFIL.EQ. 6.0D0) WRITE(IOUT,806) I,X(I)
             IF(PROFIL.EQ. 7.0D0) WRITE(IOUT,807) I,X(I)
             IF(PROFIL.EQ. 8.0D0) WRITE(IOUT,808) I,X(I)
             IF(PROFIL.EQ. 9.0D0) WRITE(IOUT,809) I,X(I)
             IF(PROFIL.EQ.10.0D0) WRITE(IOUT,810) I,X(I)
             IF(PROFIL.EQ.11.0D0) WRITE(IOUT,811) I,X(I)
             IF(PROFIL.EQ.12.0D0) WRITE(IOUT,812) I,X(I)
             IF(PROFIL.EQ.13.0D0) WRITE(IOUT,813) I,X(I)
             IF(PROFIL.EQ.14.0D0) WRITE(IOUT,814) I,X(I)
             IF(PROFIL.EQ.15.0D0) WRITE(IOUT,815) I,X(I)
             IF(PROFIL.EQ.16.0D0) WRITE(IOUT,816) I,X(I)
             IF(PROFIL.EQ.17.0D0) WRITE(IOUT,817) I,X(I)
             IF(PROFIL.EQ.18.0D0) WRITE(IOUT,818) I,X(I)
             IF(PROFIL.EQ.19.0D0) WRITE(IOUT,819) I,X(I)
             IF(PROFIL.EQ.20.0D0) WRITE(IOUT,820) I,X(I)
             IF(PROFIL.EQ.21.0D0) WRITE(IOUT,821) I,X(I)
             IF(PROFIL.EQ.22.0D0) WRITE(IOUT,822) I,X(I)
             IF(PROFIL.EQ.90.0D0) WRITE(IOUT,890) I,X(I)
             IF(PROFIL.EQ.91.0D0) WRITE(IOUT,891) I,X(I)
             IF(PROFIL.EQ.92.0D0) WRITE(IOUT,892) I,X(I)
             IF(PROFIL.EQ.93.0D0) WRITE(IOUT,893) I,X(I)
             IF(PROFIL.LT.0.0D0) WRITE(IOUT,*) 'PROFIL-ERROR'
          ENDIF
      ENDDO
      WRITE(IOUT,47)
    4 FORMAT(' --------------------------------------------')
   41 FORMAT(' EVAL:',I9,',  TIME:',F9.2,',  IFLAG:',I4,/,
     &       ' --------------------------------------------')
   42 FORMAT(' F(X) =',F38.15)
   82 FORMAT(' F(X) =',D38.6)
  142 FORMAT(' --------------------------------------------')
  183 FORMAT(' VIOLATION OF G(X)',F27.12)
  184 FORMAT(' VIOLATION OF G(X)',D27.6)
  144 FORMAT(' --------------------------------------------')
   43 FORMAT(' G(',I4,') = ',F15.8,'  (EQUALITY CONSTR)')
   44 FORMAT(' G(',I4,') = ',F15.8,'  (IN-EQUAL CONSTR)')
  431 FORMAT(' G(',I4,') = ',F15.8,'  (EQUALITY CONSTR)',
     &       '  <---  INFEASIBLE  ( G NOT = 0 )')
  441 FORMAT(' G(',I4,') = ',F15.8,'  (IN-EQUAL CONSTR)',
     &       '  <---  INFEASIBLE  ( G < 0 )')
   83 FORMAT(' G(',I4,') = ',D15.2,'  (EQUALITY CONSTR)')
   84 FORMAT(' G(',I4,') = ',D15.2,'  (IN-EQUAL CONSTR)')
  831 FORMAT(' G(',I4,') = ',D15.2,'  (EQUALITY CONSTR)',
     &       '  <---  INFEASIBLE  ( G NOT = 0 )')
  841 FORMAT(' G(',I4,') = ',D15.2,'  (IN-EQUAL CONSTR)',
     &       '  <---  INFEASIBLE  ( G < 0 )')
  145 FORMAT(' --------------------------------------------',
     &                              '            BOUNDS-PROFIL    ')
  400 FORMAT(' X(',I4,') = ',F34.15,'    !   XL___________________')
  401 FORMAT(' X(',I4,') = ',F34.15,'    !   x____________________')
  402 FORMAT(' X(',I4,') = ',F34.15,'    !   _x___________________')
  403 FORMAT(' X(',I4,') = ',F34.15,'    !   __x__________________')
  404 FORMAT(' X(',I4,') = ',F34.15,'    !   ___x_________________')
  405 FORMAT(' X(',I4,') = ',F34.15,'    !   ____x________________')
  406 FORMAT(' X(',I4,') = ',F34.15,'    !   _____x_______________')
  407 FORMAT(' X(',I4,') = ',F34.15,'    !   ______x______________')
  408 FORMAT(' X(',I4,') = ',F34.15,'    !   _______x_____________')
  409 FORMAT(' X(',I4,') = ',F34.15,'    !   ________x____________')
  410 FORMAT(' X(',I4,') = ',F34.15,'    !   _________x___________')
  411 FORMAT(' X(',I4,') = ',F34.15,'    !   __________x__________')
  412 FORMAT(' X(',I4,') = ',F34.15,'    !   ___________x_________')
  413 FORMAT(' X(',I4,') = ',F34.15,'    !   ____________x________')
  414 FORMAT(' X(',I4,') = ',F34.15,'    !   _____________x_______')
  415 FORMAT(' X(',I4,') = ',F34.15,'    !   ______________x______')
  416 FORMAT(' X(',I4,') = ',F34.15,'    !   _______________x_____')
  417 FORMAT(' X(',I4,') = ',F34.15,'    !   ________________x____')
  418 FORMAT(' X(',I4,') = ',F34.15,'    !   _________________x___')
  419 FORMAT(' X(',I4,') = ',F34.15,'    !   __________________x__')
  420 FORMAT(' X(',I4,') = ',F34.15,'    !   ___________________x_')
  421 FORMAT(' X(',I4,') = ',F34.15,'    !   ____________________x')
  422 FORMAT(' X(',I4,') = ',F34.15,'    !   ___________________XU')
  490 FORMAT(' X(',I4,') = ',F34.15,'    !   WARNING: XL = XU     ')
  491 FORMAT(' X(',I4,') = ',F34.15,'  <---  *** ERROR *** (X > XU) ')
  492 FORMAT(' X(',I4,') = ',F34.15,'  <---  *** ERROR *** (X < XL) ')
  493 FORMAT(' X(',I4,') = ',F34.15,'  <---  *** ERROR *** (XL > XU)')
  800 FORMAT(' X(',I4,') = ',D34.1,'    !   XL___________________')
  801 FORMAT(' X(',I4,') = ',D34.1,'    !   x____________________')
  802 FORMAT(' X(',I4,') = ',D34.1,'    !   _x___________________')
  803 FORMAT(' X(',I4,') = ',D34.1,'    !   __x__________________')
  804 FORMAT(' X(',I4,') = ',D34.1,'    !   ___x_________________')
  805 FORMAT(' X(',I4,') = ',D34.1,'    !   ____x________________')
  806 FORMAT(' X(',I4,') = ',D34.1,'    !   _____x_______________')
  807 FORMAT(' X(',I4,') = ',D34.1,'    !   ______x______________')
  808 FORMAT(' X(',I4,') = ',D34.1,'    !   _______x_____________')
  809 FORMAT(' X(',I4,') = ',D34.1,'    !   ________x____________')
  810 FORMAT(' X(',I4,') = ',D34.1,'    !   _________x___________')
  811 FORMAT(' X(',I4,') = ',D34.1,'    !   __________x__________')
  812 FORMAT(' X(',I4,') = ',D34.1,'    !   ___________x_________')
  813 FORMAT(' X(',I4,') = ',D34.1,'    !   ____________x________')
  814 FORMAT(' X(',I4,') = ',D34.1,'    !   _____________x_______')
  815 FORMAT(' X(',I4,') = ',D34.1,'    !   ______________x______')
  816 FORMAT(' X(',I4,') = ',D34.1,'    !   _______________x_____')
  817 FORMAT(' X(',I4,') = ',D34.1,'    !   ________________x____')
  818 FORMAT(' X(',I4,') = ',D34.1,'    !   _________________x___')
  819 FORMAT(' X(',I4,') = ',D34.1,'    !   __________________x__')
  820 FORMAT(' X(',I4,') = ',D34.1,'    !   ___________________x_')
  821 FORMAT(' X(',I4,') = ',D34.1,'    !   ____________________x')
  822 FORMAT(' X(',I4,') = ',D34.1,'    !   ___________________XU')
  890 FORMAT(' X(',I4,') = ',D34.1,'    !   WARNING: XL = XU     ')
  891 FORMAT(' X(',I4,') = ',D34.1,'  <---  *** ERROR *** (X > XU) ')
  892 FORMAT(' X(',I4,') = ',D34.1,'  <---  *** ERROR *** (X < XL) ')
  893 FORMAT(' X(',I4,') = ',D34.1,'  <---  *** ERROR *** (XL > XU)')
   47 FORMAT(/,' ')
      END
      SUBROUTINE PRINT_FINAL(IFLAG,TNOW,TMAX,EVAL,MAXEVAL,
     &                       N,M,ME,X,G,F,XL,XU,RW,ACC,WRES,PARAM,IOUT)
      IMPLICIT NONE
      INTEGER IFLAG,EVAL,MAXEVAL,N,M,ME,WRES,IOUT
      DOUBLE PRECISION TNOW,TMAX,X(*),G(*),F,XL(*),XU(*),RW(*),ACC
      DOUBLE PRECISION PARAM(*)
      IF(IFLAG.EQ.1.OR.IFLAG.EQ.2)THEN
        IF(TNOW.GE.TMAX)    WRITE(IOUT,411)
        IF(EVAL.GE.MAXEVAL) WRITE(IOUT,412)
      ENDIF
      IF(IFLAG.EQ.3.OR.IFLAG.EQ.4) WRITE(IOUT,413) NINT(PARAM(4))
      IF(IFLAG.EQ.5.OR.IFLAG.EQ.6) WRITE(IOUT,414)
 411  FORMAT(/,' OPTIMIZATION FINISHED  --->  MAXTIME REACHED')
 412  FORMAT(/,' OPTIMIZATION FINISHED  --->  MAXEVAL REACHED')
 413  FORMAT(/,' OPTIMIZATION FINISHED  --->  AUTOSTOP (=',I3,')')
 414  FORMAT(/,' OPTIMIZATION FINISHED  --->  FSTOP REACHED')
      WRITE(IOUT,42)
   42 FORMAT(/,/,'         BEST SOLUTION FOUND BY MIDACO       ')
      CALL PRINT_SOLUTION( N, M, ME, X, G, F, RW(WRES),
     &                     XL, XU, ACC, EVAL, TNOW, IFLAG, IOUT)
      END
      SUBROUTINE WARNINGS_AND_ERRORS( IFLAG , IOUT )
      IMPLICIT NONE
      INTEGER IFLAG,IOUT
      IF(IFLAG.LT.100)THEN
      WRITE(IOUT,1) IFLAG
    1 FORMAT(/,' *** WARNING ***   ( IFLAG =',I6,' )',/)
      ELSE
      WRITE(IOUT,2) IFLAG
    2 FORMAT(/,/,/,' *** MIDACO INPUT ERROR ***   ( IFLAG =',I6,' )')
      ENDIF
      END
      SUBROUTINE GET_TIME( SECONDS )
      DOUBLE PRECISION SECONDS
      CALL CPU_TIME( SECONDS )
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     This subroutine forces to flush the MIDACO output to text files.
c     Some Fortran compiler (e.g. g77 and f77) do not accept the 'flush'
c     command. In case you have any problems with 'flush', you can
c     savely remove or comment the below 'flush' command.
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE FORCE_OUTPUT( IOUT )
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      INTEGER IOUT
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      FLUSH( IOUT )
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      END
c     END OF FILE
