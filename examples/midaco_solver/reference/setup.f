C
      SUBROUTINE SETUP(  IPROB,      M,    ME,     N,   NBIN,   
     /                    NINT,   NMAX,     X,    XL,     XU, 
     /                     FEX,   PNAM,  PREF )
C
C*********************************************************************
C
C          TESTPROBLEMS FOR NONLINEAR MIXED-INTEGER OPTIMIZATION
C
C                          - INITIALIZATION -
C
C***********************************************************************
C
      INTEGER      IPROB, N, M, ME, NCONT, NINT, NBIN, NMAX, I, J 
      REAL*8       X(NMAX), XL(NMAX), XU(NMAX), FEX,tic,iout
      CHARACTER*30 PNAM, PREF
      REAL*8       R92, C92, V92, W92, CC92, VV92, WW92, 
     /             R_CROP, A1_CROP, A2_CROP, A3_CROP, DELTA_CROP, 
     /             TAU_CROP, B1_CROP, B2_CROP, B3_CROP,
     /             GU1, GL1, GU2, GL2, GU3, GL3, A, B, C, D,
     /             XEX1, XEX2, XEX3, XEX4, R_NUM(900)
      COMMON      /TESTDAT/ A(100,100), B(100), C(100,100), D(100)      
      COMMON      /CROP/R92(5), C92(5), V92(5), W92(5), CC92, VV92,
     /             WW92, R_CROP(200), A1_CROP(200), A2_CROP(200),
     /             A3_CROP(200), DELTA_CROP(200), TAU_CROP(200),
     /             B1_CROP, B2_CROP, B3_CROP          
      REAL*8       AD(10,25),BD(10),CD(25,25),DD(25),PI      
      DATA         R92/0.8D0, 0.85D0, 0.9D0, 0.65D0, 0.75D0/
      DATA         C92/1.0D0, 2.0D0, 3.0D0, 4.0D0, 2.0D0/,
     /             CC92/110.0D0/
      DATA         V92/7.0D0, 7.0D0, 5.0D0, 9.0D0, 4.0D0/,
     /             VV92/175.0D0/
      DATA         W92/7.0D0, 8.0D0, 8.0D0, 6.0D0, 9.0D0/,
     /             WW92/200.0D0/
     
      REAL SCGL,FSCALE
      DATA SCGL/1.0D-2/
      DATA FSCALE/1.0D-3/
      DATA PI /3.14159265d0/
      
      REAL*8       X0(NMAX),XEX(NMAX),LGTOT
      INTEGER      DIM
C      
      IF (IPROB.GE.98.AND.IPROB.LE.100) CALL RANDOM_NUMBERS(R_NUM)
      PNAM = '                             '
      PREF = '                             '
C
      NCONT = 0
      NINT = 0
      NBIN = 0
      DO I=1,NMAX
         XU(I) = 1.0D+10
         XL(I) = -1.0D+10
      ENDDO
      GOTO (1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,
     /     21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
     /     41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
     /     61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,
     /     81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,
     /     101,102,103,104,105,106,107,108,109,110,
     /     111,112,113,114,115,116,117,118,119,120,
     /     121,122,123,124,125,126,127,128,129,130,
     /     131,132,133,134,135,136,137,138,139,140,    
     /     141,142,143,144,145,146,147,148,149,150,
     /     151,152,153,154,155,156,157,158,159,160,
     /     161,162,163,164,165,166,167,168,169,170,
     /     171,172,173,174,175,176,177,178,179,180, 
     /     181,182,183,184,185,186,187,188,189,190,   
     /     191,192,193,194,195,196,197,198,199,200,  
     /     201,202,203,204,205,206,207,208,209,210,     
     /     211,212,213,214,215,216,217,218,219,220,  
     /     221,222,223,224,225,226,227,228,229,230,
     /     231,232,233,234,235,236,237,238,239,240,
     /     241,242,243,244,245,246,247,248,249,250,
     /     251,252,253,254,255,256,257,258,259,260,
     /     261,262,263,264,265,266,267,268,269,270,
     /     271,272,273,274,275,276,277,278,279,280,     
     /     281,282,283,284,285,286,287,288,289,290,     
     /     291,292,293,294,295,296,297,298,299,300)   
     /     , IPROB
C
C   MITP1
C
    1 CONTINUE
         PNAM = 'MITP1'
         NCONT = 2
         NINT = 3
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 1
         ME = 0
         DO I=NCONT+1,N
            X(I) = 0.5D0
            XL(I) = 0.0D0
            XU(I) = 1.0D0
         ENDDO
         X(1) = -5.0D0
         X(2) = 3.0D0
         XL(1) = -100.0D0
         XL(2) = -150.0D0
         XU(1) = 6.0D0
         XU(2) = 10.0D0
         X(3) = 5.0D0
         X(4) = -10.0D0
         X(5) = 50.0D0
         DO I=NCONT+1,N
            XL(I) = -100.0D0
            XU(I) = 100.0D0
         ENDDO
         FEX = -0.10009690D+05
      GOTO 9999
C
C   MITP2
C
    2 CONTINUE
         PNAM = 'MITP2 (QIP1)'
         PREF = '_cite{Flou99}'
         NCONT = 0
         NINT = 4
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 4
         ME = 0
         DO I=1,N
            XL(I) = -1.0D0
            XU(I) = 1.0D0
            X(I) = 0.0D0
         ENDDO
         FEX = -20.0D0
      GOTO 9999
C
C   MITP3
C
    3 CONTINUE
         PNAM = 'MITP3'
         NCONT = 2
         NINT = 0
         NBIN = 3
         N = NINT + NCONT + NBIN
         M = 7
         ME = 0
         DO I=1,NCONT
            X(I) = 2.0D0
            XL(I) = 0.0D0
            XU(I) = 4.0D0
         ENDDO
         DO I=NCONT+1,N
            X(I) = 1.0D0
            XL(I) = 0.0D0
            XU(I) = 1.0D0
         ENDDO
         FEX = 3.5D0
      GOTO 9999
C
C   MITP4
C
    4 CONTINUE
         PNAM = 'MITP4 (Asaadi 1/1)'
         PREF = '_cite{Asa73}'
         NCONT = 1
         NINT = 3
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 3
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 10.0D0
            X(I) = 5.0D0
         ENDDO
         FEX = -0.40956609D+02
      GOTO 9999
C
C   MITP5
C
    5 CONTINUE
         PNAM = 'MITP5 (Asaadi 1/2)'
         PREF = '_cite{Asa73}'
         NCONT = 0
         NINT = 4
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 3
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 10.0D0
            X(I) = 5.0D0
         ENDDO
         FEX = -38.0D0
      GOTO 9999
C
C   MITP6
C
C GEAENDERT
    6 CONTINUE
         PNAM = 'MITP6 (Asaadi 2/1)'
         PREF = '_cite{Asa73}'
         NCONT = 3
         NINT = 4
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 4
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 5.0D0
            X(I) = 2.5D0
         ENDDO
         XL(1) = -4.64D-4
         FEX = 0.69490268D+03
      GOTO 9999
C
C   MITP7
C
    7 CONTINUE
         PNAM = 'MITP7 (Asaadi 2/2)'
         PREF = '_cite{Asa73}'
         NCONT = 0
         NINT = 7
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 4
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 8.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = 700.0D0
      GOTO 9999
C
C   MITP8
C
    8 CONTINUE
         PNAM = 'MITP8 (Asaadi 3/1)'
         PREF = '_cite{Asa73}'
         NCONT = 4
         NINT = 6
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 8
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 10.0D0
            X(I) = 5.0D0
         ENDDO
         FEX = 0.37219540D+02
      GOTO 9999
C
C   MITP9
C
    9 CONTINUE
         PNAM = 'MITP9 (Asaadi 3/2)'
         PREF = '_cite{Asa73}'
         NCONT = 0
         NINT = 10
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 8
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 10.0D0
            X(I) = 5.0D0
         ENDDO
         FEX = 43.0D0
      GOTO 9999
C
C   MITP10
C
   10 CONTINUE
         PNAM = 'MITP10 (DIRTY)'
         NCONT = 12
         NINT  = 13         
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 10
         ME    = 0         
         
         CALL DIRTYDATA(AD,BD,CD,DD)
         
         DO I=1,N
            DO J=1,N
               C(I,J) = CD(I,J)
            ENDDO
            D(I) = DD(I)
         ENDDO
         DO J=1,M
            DO I=1,N
               A(J,I) = AD(J,I)
              ENDDO
            B(J) = BD(J)
         ENDDO              

         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D1
            X(I) = 0.5D0*(XU(I) + XL(I))
         ENDDO
         FEX =-304723942.920279D0
      GOTO 9999
C
C   MITP11
C
   11 CONTINUE
         PNAM = 'MITP11 (van de Braak 1)'
         PREF = '_cite{Braa01}'
         NCONT = 4
         NINT = 3
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 2
         ME = 0
         DO I=1,N
            XL(I) = -100.0D0
            XU(I) = 100.0D0
            X(I) = 0.0D0
         ENDDO
         FEX = 1.0D0
      GOTO 9999
C
C   MITP12
C
   12 CONTINUE
         PNAM = 'MITP12 (van de Braak 2)'
         PREF = '_cite{Braa01}'
         NCONT = 4
         NINT = 3
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 4
         ME = 0
         DO I=1,N
            XL(I) = -100.0D0
            XU(I) = 100.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = -0.27182811D+01
      GOTO 9999
C
C   MITP13
C
   13 CONTINUE
         PNAM = 'MITP13 (van de Braak 3)'
         PREF = '_cite{Braa01}'
         NCONT = 4
         NINT = 3
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 4
         ME = 0
         DO I=1,N
            XL(I) = -100.0D0
            XU(I) = 100.0D0
         ENDDO
         X(1) = -10.0D0
         X(2) = -20.0D0
         X(3) = 35.0D0
         X(4) = 50.0D0
         X(5) = -10.0D0
         X(6) = -20.0D0
         X(7) = -20.0D0
         FEX = -0.89800027D+07 
      GOTO 9999
C
C   MITP14
C
   14 CONTINUE
         PNAM = 'MITP14 (2DEX)'
         PREF = '_cite{CM89}'
         NCONT = 0
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 2
         ME = 0
         X(1) = 10.0D0
         XL(1) = 0.0D0
         XU(1) = 20.0D0
         X(2) = 12.0D0
         XL(2) = 12.0D0
         XU(2) = 20.0D0
         FEX = -56.9375
      GOTO 9999
C
C   MITP15
C
   15 CONTINUE
         PNAM = 'MITP15 (CROP)'
         PREF = '_cite{SRL06}'
         NCONT = 0
         NINT = 5
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 3
         ME = 0
         DO I = 1,N
            XL(I) = 1.0D0
            XU(I) = 5.0D0
            X(I) = 5.0D0
         ENDDO
         FEX = 0.10040913
      GOTO 9999
C
C   MITP16
C
   16 CONTINUE
         PNAM = 'MITP16 (TP83)'
         PREF = '_cite{HS81}'
         NCONT = 3
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 6
         ME = 0
         X(1) = 27.0D0
         X(2) = 33.0D0
         X(3) = 27.0D0
         X(4) = 27.0D0
         X(5) = 78.0D0
         XL(5) = 78.0D0
         XL(2) = 33.0D0
         XL(3) = 27.0D0
         XL(4) = 27.0D0
         XL(1) = 27.0D0
         XU(5) = 102.0D0
         XU(2) = 45.0D0
         XU(3) = 45.0D0
         XU(4) = 45.0D0
         XU(1) = 45.0D0
         FEX = -0.306655386717D+05
       GOTO 9999
C
C   MITP17
C
   17 CONTINUE
         PNAM = 'MITP17 (WP02)'
         PREF = '_cite{WP02}'
         NCONT = 1
         NINT = 1
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 2
         ME = 0
         X(1) = 1.0D0
         XL(1) = 1.0D0
         XU(1) = 8.0D0
         X(2) = 1.0D0
         XL(2) = 1.0D0
         XU(2) = 8.0D0
         FEX = -2.4444D0
      GOTO 9999
C
C   MITP18
C  
   18 CONTINUE
         PNAM = 'MITP18 (nvs01)'
         PREF = '_cite{MINLPLib}'
         NCONT = 1
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 3
         ME = 1
         X(1) = 100.0
         XL(1) = 0.0
         XU(1) = 100.0
         DO I=NCONT+1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 100.0D0
         ENDDO
         FEX = 12.46967D0
      GOTO 9999
C
C   MITP19
C
   19 CONTINUE
         PNAM = 'MITP19 (nvs02)'
         PREF = '_cite{MINLPLib}'
         NCONT = 3
         NINT = 5
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 3
         ME = 3
         XL(1) = 0.0D0
         XU(1) = 92.0D0
         X(1) = 50.0D0
         XL(2) = 90.0D0
         XU(2) = 110.0D0
         X(2) = 100.0D0          
         XL(3) = 20.0D0
         XU(3) = 25.0D0
         X(3) = 20.0D0
         DO I=NCONT+1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 100.0D0
         ENDDO
         FEX = 5.964185D0
      GOTO 9999
C
C   MITP20
C
   20 CONTINUE
         PNAM = 'MITP20 (nvs03)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 2
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 100.0D0
         ENDDO
         FEX = 16.0D0
      GOTO 9999
C
C   MITP21
C
   21 CONTINUE
         PNAM = 'MITP21 (nvs04)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 0
         ME = 0
         DO I=1,N
            XL(I) = 1.0D0
            XU(I) = 200.0D0
            X(I) = 100.0D0
         ENDDO
         FEX = 0.72D0
      GOTO 9999
C
C   MITP22
C
   22 CONTINUE
         PNAM = 'MITP22 (nvs05)'
         PREF='_cite{MINLPLib}'
         NCONT = 6
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 9
         ME = 4
         DO I=1,NCONT
            X(I) = 1.0D0
            XL(I) = 0.01D0
            XU(I) = 100.0D0
         ENDDO
         DO I=NCONT+1,N
            X(I) = 1.0D0
            XL(I) = 1.0D0
            XU(I) = 10.0D0
         ENDDO         
         XL(1) = 0.1D0
         XL(2) = 0.1D0
         XU(3) = 5.0D0
         X(5) = 2.0D0
         FEX = 5.470934D0
      GOTO 9999
C
C   MITP23
C
   23 CONTINUE
         PNAM = 'MITP23 (nvs06)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 0
         ME = 0
         DO I=1,N
            XL(I) = 1.0D0
            XU(I) = 200.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = 1.770312D0
      GOTO 9999
C
C   MITP24
C
   24 CONTINUE
         PNAM = 'MITP24 (nvs07)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 3
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 2
         ME = 0
         X(1) = 1.0D0
         X(2) = 1.0D0
         X(3) = 1.0D0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
         ENDDO
         FEX = 4.0D0
      GOTO 9999
C
C   MITP25
C
   25 CONTINUE
         PNAM = 'MITP25 (nvs08)'
         PREF = '_cite{MINLPLib}'
         NCONT = 1
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 3
         ME = 0
           XL(1) = 0.001D0
           XU(1) = 200.0D0
         X(1) = 1.0D0
         X(2) = 1.0D0
         X(3) = 1.0D0
         DO I=NCONT+1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
         ENDDO
         FEX = 23.44973D0 
      GOTO 9999
C
C   MITP26
C
   26 CONTINUE
         PNAM = 'MITP26 (nvs09)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 10
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 0
         ME = 0
         DO I=1,N
            XL(I) = 3.0D0
            XU(I) = 9.0D0
            X(I) = 5.0D0
         ENDDO         
         FEX = -43.13434D0 
      GOTO 9999
C
C   MITP27
C
   27 CONTINUE
         PNAM = 'MITP27 (nvs10)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 2
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = -310.8D0
      GOTO 9999
C
C   MITP28
C
   28 CONTINUE
         PNAM = 'MITP28 (nvs11)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 3
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 3
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = -431.0D0
      GOTO 9999
C
C   MITP29
C
   29 CONTINUE
         PNAM = 'MITP29 (nvs12)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 4
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 4
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = -481.2D0
      GOTO 9999
C
C   MITP30
C
   30 CONTINUE
         PNAM = 'MITP30 (nvs13)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 5
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 5
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = -585.2D0
      GOTO 9999
C
C   MITP31
C
   31 CONTINUE
         PNAM = 'MITP31 (nvs14)'
         PREF = '_cite{MINLPLib}'
         NCONT = 3
         NINT = 5
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 3
         ME = 3
         XL(1) = 0.0D0
         XU(1) = 92.0D0
         X(1) = 46.0D0
         XL(2) = 90.0D0
         XU(2) = 110.0D0
         X(2) = 100.0D0
         XL(3) = 20.0D0
         XU(3) = 25.0D0
         X(3) = 23.0D0
         DO I=NCONT+1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 100.0D0
         ENDDO
         XU(4) = 1.0D1
         X(4) = 5.0D0
         XU(5) = 2.0D1
         X(5) = 10.0D0
         XU(6) = 2.0D1
         X(6) = 10.0D0
         FEX = -40358.15D0
      GOTO 9999
C
C   MITP32
C
   32 CONTINUE
         PNAM = 'MITP32 (nvs15)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 3
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 1
         ME = 0
         DO I=NCONT+1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 100.0D0
         ENDDO
         FEX = 1.0D0
      GOTO 9999
C
C   MITP33
C
   33 CONTINUE
         PNAM = 'MITP33 (nvs16)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 0
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = 0.703125D0    
      GOTO 9999
C
C   MITP34
C
   34 CONTINUE
         PNAM = 'MITP34 (nvs17)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 7
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 7
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = -1100.4D0
      GOTO 9999
C
C   MITP35
C
   35 CONTINUE
         PNAM = 'MITP35 (nvs18)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 6
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 6
         ME = 0
         DO I=1,N
            XL(I) = 1.0D0
            XU(I) = 200.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = -778.4D0
      GOTO 9999
C
C   MITP36
C
   36 CONTINUE
         PNAM = 'MITP36 (nvs19)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 8
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 8
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = -1098.4D0
      GOTO 9999
C
C   MITP37
C
   37 CONTINUE
         PNAM = 'MITP37 (nvs20)'
         PREF = '_cite{MINLPLib}'
         NCONT = 11
         NINT = 5
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 8
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 2.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = 0.23092217D+3
      GOTO 9999
C
C   MITP38
C
   38 CONTINUE
         PNAM = 'MITP38 (nvs21)'
         PREF = '_cite{MINLPLib}'
         NCONT = 1
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 2
         ME = 0
         DO I=1,N
            XL(I) = 1.0D0
            XU(I) = 200.0D0
            X(I) = 1.0D0
         ENDDO
         XL(1) = 0.0D0
         XU(1) = 0.2D0
         X(1) = 0.1D0
         FEX = -5.684783D0
      GOTO 9999
C
C   MITP39
C
   39 CONTINUE
         PNAM = 'MITP39 (nvs22)'
         PREF = '_cite{MINLPLib}'
         NCONT = 4
         NINT = 4
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 9
         ME = 4
         DO I=1,N
            XL(I) = 0.01D0
            XU(I) = 100.0D0
            X(I) = 1.0D0
         ENDDO
         XL(5) = 1.0D0
         XL(6) = 1.0D0
         XL(7) = 1.0D0
         XL(8) = 1.0D0
         X(5) = 2.0D0
         X(6) = 2.0D0
         X(7) = 2.0D0
         X(8) = 2.0D0
         XU(5) = 200.0D0
         XU(6) = 200.0D0
         XU(7) = 20.0D0
         XU(8) = 20.0D0
         FEX    = 6.05822D0
      GOTO 9999
C
C   MITP40
C
   40 CONTINUE
         PNAM = 'MITP40 (nvs23)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 9
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 9
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 100.0D0
         ENDDO
         FEX = -1125.2D0
      GOTO 9999
C
C   MITP41
C
   41 CONTINUE
         PNAM = 'MITP41 (nvs24)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 10
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 10
         ME = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 100.0D0
         ENDDO
         FEX = -1033.2D0
      GOTO 9999
C
C   MITP42
C
   42 CONTINUE
         PNAM = 'MITP42 (GEAR1)'
         PREF = '_cite{MINLPLib}'
         NCONT = 4
         NINT = 4
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 4
         ME = 4
         DO I=1,N
            XL(I) = 12.0D0
            XU(I) = 60.0D0
            X(I) = 36.0D0
         ENDDO
         FEX = 1.0D0
      GOTO 9999
C
C   MITP43
C
   43 CONTINUE
         PNAM = 'MITP43 (WINDFAC)'
         PREF = '_cite{MINLPLib}'
         NCONT = 11
         NINT = 3
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 13
         ME = 13
         DO I=1,N
            XL(I) = -100.0D0
            XU(I) = 100.0D0
            X(I) = 10.0D0
         ENDDO
         X(1) = 1.5D0
         X(2) = 1.0D0
         X(11) = 0.8D0  
         X(12) = 1.D0  
         X(13) = 15.D0  
         X(14) = 3.D0 
         XL(12) = 1.0D0
         XL(13) = 1.0D0
         XL(14) = 1.0D0
         XU(12) = 10.0D0
         XU(13) = 100.0D0
         XU(14) = 100.0D0
         XL(11) = 0.8D0
         FEX = 25.44873D-2
      GOTO 9999
C
C   MITP44
C
   44 CONTINUE
         PNAM = 'MITP44 (Duran/Grossmann 1)'
         PREF = '_cite{DG86}'
         NCONT = 3
         NINT = 0
         NBIN = 3
         N = NINT + NCONT + NBIN
         M = 6
         ME = 0
         DO I=1,NCONT
            X(I) = 0.5D0
            XL(I) = 0.0D0
            XU(I) = 2.0D0
         ENDDO
         XU(3) = 1.0
         DO I=NCONT+1,N
            X(I) = 0.5D0
            XL(I) = 0.0D0
            XU(I) = 1.0D0
         ENDDO
         FEX = 6.00974173126459D0
      GOTO 9999
C
C   MITP45
C
   45 CONTINUE
         PNAM = 'MITP45 (Duran/Grossmann 2)'
         PREF = '_cite{DG86}'
         NCONT = 6
         NINT = 0
         NBIN = 5
         N = NINT + NCONT + NBIN
         M = 14
         ME = 1
         DO I=1,NCONT
            X(I) = 1.0D0
            XL(I) = 0.0D0
            XU(I) = 2.0D0
         ENDDO
         XU(4) = 1.0D+1
         XU(5) = 1.0D+1
         XU(6) = 3.0D0
         DO I=NCONT+1,N
            X(I) = 0.5D0
            XL(I) = 0.0D0
            XU(I) = 1.0D0
         ENDDO
         FEX = 0.73035665D+02
      GOTO 9999
C
C   MITP46
C
   46 CONTINUE
         PNAM = 'MITP46 (Duran/Grossmann 3)'
         PREF = '_cite{DG86}'
         NCONT = 9
         NINT = 0
         NBIN = 8
         N = NINT + NCONT + NBIN
         M = 23
         ME = 2
         DO I=1,NCONT
            X(I) = 1.0D0
            XL(I) = 0.0D0
            XU(I) = 2.0D0
         ENDDO
         XU(3) = 1.0D0
         XU(8) = 1.0D0
         XU(9) = 3.0D0
         DO I=NCONT+1,N
            X(I) = 0.0D0
            XL(I) = 0.0D0
            XU(I) = 1.0D0
         ENDDO
         FEX = 68.01D0
      GOTO 9999
C
C   MITP47
C
   47 CONTINUE
         PNAM = 'MITP47 (Floudas 1)'
         PREF = '_cite{Flou99}'
         NCONT = 2
         NINT = 0
         NBIN = 3
         N = NINT + NCONT + NBIN
         M = 5
         ME = 2
         DO I=1,NCONT
            X(I) = 1.0
            XL(I) = 0.0001
            XU(I) = 1.0D+1
         ENDDO
         DO I=NCONT+1,N
            X(I) = 0.5D0
            XL(I) = 0.0D0
            XU(I) = 1.0D0
         ENDDO
         FEX = 0.76671801D+01
      GOTO 9999
C
C   MITP48
C
   48 CONTINUE
         PNAM = 'MITP48 (Floudas 2)'
         PREF = '_cite{Flou99}'
         NCONT = 2
         NINT = 0
         NBIN = 1
         N = NINT + NCONT + NBIN
         M = 3
         ME = 0
         X(1) = 0.2D0
         XL(1) = 0.2D0
         XU(1) = 1.0D0
         X(2) = -1.5D0
         XL(2) = -2.22554D0
         XU(2) = -1.0D0
         DO I=NCONT+1,N
            X(I) = 0.5D0
            XL(I) = 0.0D0
            XU(I) = 1.0D0
         ENDDO
         FEX = 0.107654D+01
      GOTO 9999
C
C   MITP49
C
   49 CONTINUE
         PNAM = 'MITP49 (Floudas 3)'
         PREF = '_cite{Flou99}'
         NCONT = 3
         NINT = 0
         NBIN = 4
         N = NINT + NCONT + NBIN
         M = 9
         ME = 0
         DO I=1,NCONT
            X(I) = 1.0D0
            XL(I) = 0.0D0
            XU(I) = 1.0D+7
         ENDDO
         DO I=NCONT+1,N
            X(I) = 0.5D0
            XL(I) = 0.0D0
            XU(I) = 1.0D0
         ENDDO
         FEX = 0.45795825D+01
      GOTO 9999
C
C   MITP50
C
   50 CONTINUE
         PNAM = 'MITP50 (Floudas 4)'
         PREF = '_cite{Flou99}'
         NCONT = 3
         NINT = 0
         NBIN = 8
         N = NINT + NCONT + NBIN
         M = 7
         ME = 3
         DO I=1,NCONT
            X(I) = 0.5D0
            XL(I) = 0.0D0
            XU(I) = 0.999D0
         ENDDO
         DO I=NCONT+1,N
            X(I) = 0.5D0
            XL(I) = 0.0D0
            XU(I) = 1.0D0
         ENDDO
         FEX = -0.94347050D+0
      GOTO 9999
C
C   MITP51
C
   51 CONTINUE
         PNAM = 'MITP51 (Floudas 5)'
         PREF = '_cite{Flou99}'
         NCONT = 0
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 4
         ME = 0
         DO I=1,N
            X(I) = 2.5D0
            XL(I) = 1.0D0
            XU(I) = 5.0D0
         ENDDO
         FEX = 31.0D0
      GOTO 9999
C
C   MITP52
C
   52 CONTINUE
         PNAM = 'MITP52 (Floudas 6)'
         PREF = '_cite{Flou99}'
         NCONT = 1
         NINT = 1
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 3
         ME = 0
         X(1) = 5.0D0
         XL(1) = 1.0D0
         XU(1) = 10.0D0
         X(2) = 3.0D0
         XL(2) = 1.0D0
         XU(2) = 6.0D0
         FEX = -17.0D0
      GOTO 9999
C
C   MITP53
C
   53 CONTINUE
         PNAM = 'MITP53 (OAER)'
         PREF = '_cite{MINLPLib}'
         NCONT = 6
         NINT = 0
         NBIN = 3
         N = NINT + NCONT + NBIN
         M = 7
         ME = 3
         DO I = 1,NCONT
            XL(I) = 0.0D0
            XU(I) = 10.0D0
            X(I) = 5.0D0
         ENDDO        
         DO I = NCONT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO        
         FEX = -1.923099D0        
      GOTO 9999
C
C   MITP54
C
   54 CONTINUE
         PNAM = 'MITP54 (SPRING)'
         PREF = '_cite{MINLPLib}'
         NCONT = 6
         NINT = 1
         NBIN = 11
         N = NINT + NCONT + NBIN
         M = 8
         ME = 5
         DO I = 1, NCONT 
            XL(I) = 0.0
            XU(I) = 10.0
            X(I) = 1.0D0
         ENDDO
         DO I = 1, NCONT + NINT
            XL(I) = 0.0
            XU(I) = 10.0
            X(I) = 1.0D0
         ENDDO
         DO I=NCONT+NINT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 1.0D0
         ENDDO
         XL(1) = 0.414D0
         XL(2) = 0.207D0
         XL(3) = 0.00178571428571429D0 
         X(3) = 0.002D0 
         XU(3) = 0.02D0
         XL(4) = 1.1D0 
         X(4) = 2.0D0 
         XL(5) = 1.0D-1 
         XU(5) = 9.5D0
         XL(6) = 5.0D0     
         X(6) = 7.0D0
         XU(6) = 10.0D0
         FEX = 0.8462457D0
      GOTO 9999
C
C   MITP55
C
   55 CONTINUE
         PNAM = 'MITP55 (GEAR)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 4
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 0
         ME = 0
         DO I = 1, N
            XL(I) = 12.0D0
            XU(I) = 60.0D0
            X(I) = 20.0D0
         ENDDO
         FEX = 1.0D0
      GOTO 9999
C
C   MITP56
C
   56 CONTINUE
         PNAM = 'MITP56 (DAKOTA)'
         PREF = '_cite{EGW02}'
         NCONT = 2
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 2
         ME = 0
         DO I = 1, NCONT
            XL(I) = -10.0D0
            XU(I) = 10.0D0
         ENDDO
         X(1) = 0.5D0
         X(2) = 1.5D0
         DO I = NCONT+1, N
            XL(I) = 0.0D0
            XU(I) = 4.0D0
            X(I) = 2.0D0
         ENDDO
         XEX1 = 0.5D0
         XEX2 = 0.5D0
         XEX3 = 1.0D0
         XEX4 = 1.0D0
         FEX = (XEX1 - 1.4D0)**4 + (XEX2 - 1.4D0)**4 
     /              + (XEX3 - 1.4D0)**4 + (XEX4 - 1.4D0)**4
      GOTO 9999
C
C   MITP57
C
   57 CONTINUE
         PNAM = 'MITP57 (GEAR4)'
         PREF = '_cite{MINLPLib}'
         NCONT = 2
         NINT = 4
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 1
         ME = 1
         DO I = 1, NCONT
            XL(I) = 0.0D0
            XU(I) = 100.0D0
            X(I) = 1.0D0
         ENDDO
         DO I = NCONT+1, N
            XL(I) = 12.0D0
            XU(I) = 60.0D0
            X(I) = 20.0D0
         ENDDO
         FEX = 1.0D0
      GOTO 9999         
C
C   MITP58
C
   58 CONTINUE
         PNAM = 'MITP58 (GEAR3)'
         PREF = '_cite{MINLPLib}'
         NCONT = 4
         NINT = 4
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 4
         ME = 4
         DO I = 1,N
            XL(I) = 12.0D0
            XU(I) = 60.0D0
            X(I) = 20.0D0
         ENDDO
         FEX = 1.0D0
      GOTO 9999      
C
C   MITP59
C
   59 CONTINUE
         PNAM = 'MITP59 (EX1252A)'
         PREF = '_cite{MINLPLib}'
         NCONT = 15
         NINT = 6
         NBIN = 3
         N = NINT + NCONT + NBIN
         M = 34
         ME = 13
         DO I = 1,NCONT
            XL(I) = 0.0D0
            XU(I) = 10000.0D0
            X(I) = 0.2D0
         ENDDO
         DO I = NCONT+1,NCONT+NINT
            XL(I) = 0.0D0
            XU(I) = 200.0D0
            X(I) = 1.0D0
         ENDDO
         DO I = NCONT+NINT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 1.0D0
         ENDDO
         XU(1) = 80.0D0
         XU(2) = 25.0D0
         XU(3) = 45.0D0 
         XU(4) = 0.295D4
         X(4) = 0.0983333333333333D0
         X(4) = 1000.0D0
         XU(5) = 0.295D4
         X(5) = 0.0983333333333333D0
         X(5) = 1000.0D0
         XU(6) = 0.295D4
         X(6) = 0.0983333333333333D0
         X(6) = 1000.0D0
         XU(7) = 400.0D0
         X(7) = 133.333333333333D0
         XU(8) = 400.0D0
         X(8) = 133.333333333333D0
         XU(9) = 400.0D0
         X(9) = 133.333333333333D0
         XU(10) = 350.0D0
         X(10) = 116.666666666667D0
         XU(11) = 350.0D0
         X(11) = 116.666666666667D0
         XU(12) = 350.0D0
         X(12) = 116.666666666667D0
         XL(10) = 1.0D0
         XL(11) = 1.0D0
         XL(12) = 1.0D0
         XU(13) = 1.0D0
         X(13) = 0.33333333333333D0
         XU(14) = 1.0D0
         X(14) = 0.33333333333333D0
         XU(15) = 1.0D0
         X(15) = 0.33333333333333D0
         XU(16) = 3.0D0
         XU(17) = 3.0D0
         XU(18) = 3.0D0
         XU(19) = 3.0D0
         XU(20) = 3.0D0
         XU(21) = 3.0D0
         XL(19) = 1.0D0
         XL(20) = 1.0D0
         XL(21) = 1.0D0
         XU(22) = 1.0D0
         XU(23) = 1.0D0
         XU(24) = 1.0D0
         XL(22) = 0.0D0
         XL(23) = 0.0D0
         XL(24) = 0.0D0
         X(22) = 1.0D0
         X(23) = 1.0D0
         X(24) = 1.0D0
c         FEX = 0.16620938D+06
c         FEX = 0.14457706D+06
         FEX = 128918.0D0
      GOTO 9999 
C
C   MITP60
C
   60 CONTINUE
         PNAM = 'MITP60 (EX1263A)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 20
         NBIN = 4
         N = NINT + NBIN + NCONT
         M = 35
         ME = 0
         DO I = 1,NINT
            XL(I) = 0.0D0
            XU(I) = 5.0D0
            X(I) = 1.0D0
         ENDDO
         DO I = NINT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         DO I=17,20
            XU(I) = 30.0D0
            X(I) = 10.0D0
         ENDDO   
         FEX = 19.6D0
      GOTO 9999  
C
C   MITP61
C
   61 CONTINUE
         PNAM = 'MITP61 (EX1264A)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 20
         NBIN = 4
         N = NINT + NCONT + NBIN
         M = 35
         ME = 0
         DO I = 1,NINT
            XL(I) = 0.0D0
            XU(I) = 5.0D0
            X(I) = 1.0D0
         ENDDO
         DO I = NINT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         XU(17) = 15.0D0
         XU(18) = 12.0D0
         XU(19) = 9.0D0
         XU(20) = 6.0D0
         FEX = 8.6D0
      GOTO 9999              
C
C   MITP62
C
   62 CONTINUE
         PNAM = 'MITP62 (EX1265A)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 30
         NBIN = 5
         N = NINT + NCONT + NBIN
         M = 44
         ME = 0
         DO I = 1,NINT
            XL(I) = 0.0D0
            XU(I) = 5.0D0
            X(I) = 1.0D0
         ENDDO
         XU(26) = 15.0D0
         XU(27) = 12.0D0
         XU(28) = 9.0D0
         XU(29) = 6.0D0
         XU(30) = 6.0D0
         DO I = NINT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         FEX = 10.3D0
      GOTO 9999
C
C   MITP63
C
   63 CONTINUE
         PNAM = 'MITP63 (EX1266A)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 42
         NBIN = 6
         N = NINT + NCONT + NBIN
         M = 53
         ME = 0
         DO I = 1,NINT
            XL(I) = 0.0D0
            XU(I) = 5.0D0
            X(I) = 0.0D0
         ENDDO
         DO I=NINT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         X(1) = 1.0D0
         X(7) = 2.0D0
         X(14) = 2.0D0
         X(20) = 1.0D0
         X(26) = 2.0D0
         X(31) = 1.0D0         
         XU(37) = 15.0D0
         X(37) = 8.0D0
         XU(38) = 12.0D0
         X(38) = 7.0D0
         XU(39) = 8.0D0
         XU(40) = 7.0D0
         XU(41) = 4.0D0
         XU(42) = 2.0D0
         FEX = 16.3D0
      GOTO 9999  
C
C   MITP64
C
   64 CONTINUE
         PNAM = 'MITP64 (DU__OPT5)'
         PREF = '_cite{MINLPLib}'
         NCONT = 7
         NINT = 13
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 9
         ME = 0
         DO I = 1,N
            XL(I) = 0.0D0
            XU(I) = 10.0D0
            X(I) = 1.0D0
         ENDDO
         XL(1) = -0.0408D0
         X(1) = -0.0288D0
         XU(1) = -0.0288D0
         X(2) = 0.0D0
         XU(2) = 0.008D0
         XL(3) = -0.0311D0
         X(3) = -0.0211D0
         XU(3) = -0.0211D0
         XL(4) = 0.1D0
         X(4) = 0.5D0
         XU(4) = 1.0D0
         XL(5) = 0.01D0
         X(5) = 0.05D0
         XU(5) = 0.08D0
         XL(6) = 0.1D0
         X(6) = 0.5D0
         XU(6) = 1.0D0
         XL(7) = 0.01D0
         X(7) = 0.05D0
         XU(7) = 0.08D0
         XU(8) = 3.0D0
         XL(9) = 9.0D0
         X(9) = 9.0D0
         XU(9) = 9.0D0
         XU(10) = 42.0D0
         XL(11) = 11.0D0
         X(11) = 15.0D0         
         XU(11) = 21.0D0
         XU(12) = 2.0D0
         XU(13) = 2.0D0
         XU(14) = 2.0D0
         XU(17) = 5.0D0
         XU(18) = 16.0D0
         XU(19) = 16.0D0
         XU(20) = 8.0D0
c         FEX = 0.84806354D+01
c         FEX = 0.12187289D+02
         FEX =  8.3435835598D+00
      GOTO 9999
C
C   MITP65
C
   65 CONTINUE
         PNAM = 'MITP65 (DU__OPT)'
         PREF = '_cite{MINLPLib}'
         NCONT = 7
         NINT = 13
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 9
         ME = 0
         DO I = 1, N
            XL(I) = 0.0D0
            XU(I) = 1000.0D0
            X(I) = 1.0D0
         ENDDO
         XL(1) = -0.0408D0
         XU(1) = -0.0288D0
         X(1) = -0.0288D0
         XU(2) = 0.008D0
         X(2) = 0.0D0
         XL(3) = -0.0311D0
         XU(3) = -0.0211D0
         X(3) = -0.0211D0
         XL(4) = 0.1D0
         XU(4) = 1.0D0
         X(4) = 0.5D0
         XL(5) = 0.01D0
         XU(5) = 0.08D0
         X(5) = 0.05
         XL(6) = 0.1D0
         XU(6) = 1.0D0
         X(6) = 0.5D0
         XL(7) = 0.01D0
         X(7) = 0.05D0
         XU(7) = 0.08D0
         X(8) = 0.05D0
         XU(8) = 16.0D0
         XL(9) = 40.0D0
         X(9) = 46.0D0
         XU(9) = 350.0D0
         X(10) = 100.0D0
         XU(10) = 2500.0D0
         XL(11) = 51.0D0 
         X(11) = 75.0D0
         XU(11) = 108.0D0
         XU(12) = 10.0D0
         XU(13) = 10.0D0
         XU(14) = 10.0D0
         XU(15) = 25.0D0
         XU(16) = 10.0D0
         XU(17) = 50.0D0
         XU(18) = 80.0D0
         XU(19) = 80.0D0
         XU(20) = 40.0D0
c         FEX = 0.53152727D+01
c         FEX = 3.5392D+00     
         FEX = 3.2200349D+00    
      GOTO 9999
C
C   MITP66
C
   66 CONTINUE
         PNAM = 'MITP66 (ST__E32)'
         PREF = '_cite{MINLPLib}'
         NCONT = 16
         NINT = 19
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 18
         ME = 17
         DO I = 1,N
            XL(I) = 0.0D0
            XU(I) = 14.0D0
            X(I) = 1.0D0
         ENDDO
         XL(1) = 1.0D0
         X(1) = 250.0D0
         XU(1) = 1000.0D0
         XL(2) = 1.0D0
         X(2) = 500.0D0
         XU(2) = 1000.0D0
         XL(3) = 1.0D0
         X(3) = 20.0D0
         XU(3) = 100.0D0
         XL(4) = 1.0D0
         X(4) = 15.0D0
         XU(4) = 32.2D0
         XL(5) = 1.0D0
         X(5) = 20.0D0
         XU(5) = 100.0D0
         XL(6) = 18.4D0
         X(6) = 20.0D0
         XU(6) = 100.0D0
         XL(7) = 1.4D0
         X(7) = 2.0D0
         XL(8) = 1.4D0
         X(8) = 2.0D0
         XL(9) = 0.001D0
         X(9) = 0.5D0
         XU(9) = 1.0D0
         XL(10) = 0.001D0
         X(10) = 0.5D0
         XU(10) = 1.0D0
         XL(11) = 0.001D0
         X(11) = 0.5D0
         XU(11) = 1.0D0
         XL(12) = 0.001D0
         X(12) = 0.5D0
         XU(12) = 1.0D0
         XL(13) = 0.001D0
         X(13) = 0.5D0
         XU(13) = 1.0D0         
         XL(14) = 0.001D0
         X(14) = 5.0D0
         XU(14) = 10.0D0
         XL(15) = 0.001D0
         X(15) = 5.0D0
         XU(15) = 10.0D0
         XL(16) = -10.0D0
         X(16) = 1.0D0
         XU(16) = 10.0D0
         FEX = -0.14304073D+01
      GOTO 9999
C
C   MITP67
C
   67 CONTINUE
         PNAM = 'MITP67 (ST__E36)'
         PREF = '_cite{MINLPLib}'
         NCONT = 1
         NINT = 1
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 2
         ME = 1
         XL(1) = 3.0D0
         X(1) = 4.0D0
         XU(1) = 5.5D0
         XL(2) = 15.0D0
         X(2) = 20.0D0
         XU(2) = 25.0D0
         FEX = -246.0D0
      GOTO 9999
C
C   MITP68
C
   68 CONTINUE
         PNAM = 'MITP68 (ST__E38)'
         PREF = '_cite{MINLPLib}'
         NCONT = 2
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 3
         ME = 0
         XL(1) = 40.0D0
         X(1) = 60.0D0
         XU(1) = 80.0D0
         XL(2) = 20.0D0
         X(2) = 40.0D0
         XU(2) = 60.0D0
         XL(3) = 18.0D0
         X(3) = 50.0D0
         XU(3) = 100.0D0
         XL(4) = 10.0D0
         X(4) = 20.0D0
         XU(4) = 100.0D0
         FEX = 7197.72714852D0
      GOTO 9999
C
C   MITP69
C
   69 CONTINUE
         PNAM = 'MITP69 (ST__E40)'
         PREF = '_cite{MINLPLib}'
         NCONT = 1
         NINT = 3
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 5
         ME = 1
         XL(1) = 1.0D0
         X(1) = 10.0D0
         XU(1) = 100.0D0
         XL(2) = 1.0D0
         X(2) = 5.0D0
         XU(2) = 12.0D0
         XL(3) = 1.0D0
         X(3) = 5.0D0
         XU(3) = 12.0D0
         XL(4) = 1.0D0
         X(4) = 5.0D0
         XU(4) = 12.0D0
         FEX = 28.24264D0  
      GOTO 9999
C
C   MITP70
C
   70 CONTINUE
         PNAM = 'MITP70 (ST__MIQP1)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 0
         NBIN = 5
         N = NINT + NCONT + NBIN
         M = 1
         ME = 0
         DO I = 1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         FEX = 281.0D0
      GOTO 9999
C
C   MITP71
C
   71 CONTINUE
         PNAM = 'MITP71 (ST__MIQP2)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 4
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 3
         ME = 0
         DO I = 1,N
            XL(I) = 0.0D0
            X(I) = 0.0D0
         ENDDO
         XU(1) = 1.0D0
         XU(2) = 1.0D0
         XU(3) = 1.0D10
         XU(4) = 1.0D10
         FEX = 2.0D0
      GOTO 9999
C
C   MITP72
C
   72 CONTINUE
         PNAM = 'MITP72 (ST__MIQP3)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 1
         ME = 0
         DO I = 1, N
            XL(I) = 0.0D0
            X(I) = 0.0D0
         ENDDO
         XU(1) = 3.0D0
         XU(2) = 1.0D12
         FEX = -6.0D0
      GOTO 9999
C
C   MITP73
C
   73 CONTINUE
         PNAM = 'MITP73 (ST__MIQP4)'
         PREF = '_cite{MINLPLib}'
         NCONT = 3
         NINT = 0
         NBIN = 3
         N = NINT + NCONT + NBIN
         M = 4
         ME = 0
         DO I = 1, N
            XL(I) = 0.0D0
c            XU(I) = 1.0D12
            XU(I) = 1.0D6
            X(I) = 0.0D0
         ENDDO
         XU(4) = 1.0D0
         XU(5) = 1.0D0
         XU(6) = 1.0D0
         FEX = -4574.0D0
      GOTO 9999
C
C   MITP74
C
   74 CONTINUE
         PNAM = 'MITP74 (ST__MIQP5)'
         PREF = '_cite{MINLPLib}'
         NCONT = 5
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 13
         ME = 0
         DO I = 1, N
            XL(I) = -100.0D0
            XU(I) = 100.0D0
            X(I) = 0.0D0
         ENDDO
         XL(1) = -7.24380468458D0
         XU(1) = 22.6826188429D0
         XL(2) = -6.0023781122D0
         XU(2) = 3.80464419615D0
         XL(3) = -0.797166188733D0
         XU(3) = 11.5189336042D0
         XL(4) = -8.75189948987D0
         XU(4) = 14.5864991498D0
         XL(5) = 8.98296319621D-17
         XU(5) = 19.4187214575D0
         XU(6) = 2.43857786822
         XU(7) = 1.0D0
         FEX = -333.89D0
      GOTO 9999
C
C   MITP75
C
   75 CONTINUE
         PNAM = 'MITP75 (ST__TEST1)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 5
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 1
         ME = 0
         DO I = 1, N
            XL(I) = -100.0D0
            XU(I) = 100.0D0
            X(I) = 0.0D0
         ENDDO
         FEX = -4500.0
      GOTO 9999
C
C   MITP76
C
   76 CONTINUE
         PNAM = 'MITP76 (ST__TEST2)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 1
         NBIN = 5
         N = NINT + NCONT + NBIN
         M = 2
         ME = 0
         DO I = 1, N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         XU(1) = 1.0D12
         FEX = -9.25D0
      GOTO 9999
C
C   MITP77
C
   77 CONTINUE
         PNAM = 'MITP77 (ST__TEST3)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 3
         NBIN = 10
         N = NINT + NCONT + NBIN
         M = 10
         ME = 0
         DO I = 1, N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         XU(1) = 1.0E12
         XU(2) = 1.0E12
         XU(3) = 1.0E12
         FEX = -7.0D0
      GOTO 9999
C
C   MITP78
C
   78 CONTINUE
         PNAM = 'MITP78 (ST__TEST4)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 4
         NBIN = 2
         N = NINT + NCONT + NBIN
         M = 5
         ME = 0
         DO I = 1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D12
            X(I) = 1.0
         ENDDO
         XU(5) = 1.0D0
         X(5) = 0.5D0
         XU(6) = 1.0D0
         X(6) = 0.5D0
         XU(4) = 2.0D0
         FEX = -7.0D0
      GOTO 9999
C
C   MITP79
C
   79 CONTINUE
         PNAM = 'MITP79 (ST__TEST5)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 0
         NBIN = 10
         N = NINT + NCONT + NBIN
         M = 11
         ME = 0
         DO I = 1, N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         FEX = -110.0D0
      GOTO 9999
C
C   MITP80
C
   80 CONTINUE
         PNAM = 'MITP80 (ST__TEST6)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 10
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 5
         ME = 0
         DO I = 1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         FEX = 471.0D0
      GOTO 9999
C
C   MITP81
C
   81 CONTINUE
         PNAM = 'MITP81 (ST__TEST8)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 24
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 20
         ME = 0
         DO I = 1,N
            XL(I) = 0.0D0
            XU(I) = 2000.0D0
            X(I) = 100.0D0
         ENDDO
         FEX = -29605.0D0
      GOTO 9999
C
C   MITP82
C
   82 CONTINUE
         PNAM = 'MITP82 (ST__TESTGR1)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 10
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 5
         ME = 0
         DO I = 1,N
            XL(I) = 0.0D0
            XU(I) = 100.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = -12.8116D0
      GOTO 9999
C
C   MITP83
C
   83 CONTINUE
         PNAM = 'MITP83 (ST__TESTGR3)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 20
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 20
         ME = 0
         DO I = 1,N
            XL(I) = 0.0D0
            X(I) = 50.0D0
            XU(I) = 100.0D0
         ENDDO
         FEX = -20.59D0
      GOTO 9999
C
C   MITP84
C
   84 CONTINUE
         PNAM = 'MITP84 (ST__TESTPH4)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 3
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 10
         ME = 0
         DO I = 1,N
            XL(I) = 0.0D0
            XU(I) = 100.0D0
            X(I) = 1.0D0
         ENDDO
         FEX = -80.5D0
      GOTO 9999
C
C   MITP85
C
   85 CONTINUE
         PNAM = 'MITP85 (TLN2)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 6
         NBIN = 2
         N = NINT + NCONT + NBIN
         M = 12
         ME = 0
         DO I = 1,NINT
            XL(I) = 1.0D0
            XU(I) = 5.0D0
            X(I) = 1.0D0
         ENDDO
         XL(3) = 0.0
         XL(6) = 0.0
         XL(5) = 0.0
         DO I = NINT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         XU(1) = 15.0D0
         XU(2) = 15.0D0
         FEX = 2.3D+0 
      GOTO 9999
C
C   MITP86
C
   86 CONTINUE
         PNAM = 'MITP86 (TLN4)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 20
         NBIN = 4
         N = NINT + NCONT + NBIN
         M = 24
         ME = 0
         DO I = NINT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         DO I = 1,NINT
            XL(I) = 0.0D0
            XU(I) = 5.0D0
            X(I) = 1.0D0
         ENDDO
         XU(1) = 12.0D0
         XU(2) = 12.0D0
         XU(3) = 12.0D0
         XU(4) = 12.0D0
         FEX = 8.3D0
      GOTO 9999
C
C   MITP87
C
   87 CONTINUE
         PNAM = 'MITP87 (TLN5)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 30
         NBIN = 5
         N = NINT + NCONT + NBIN
         M = 30
         ME = 0
         DO I = NINT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         DO I = 1,NINT
            XL(I) = 0.0D0
            XU(I) = 5.0D0
            X(I) = 1.0D0
         ENDDO
         XU(1) = 15.0D0 
         XU(2) = 15.0D0 
         XU(3) = 15.0D0 
         XU(4) = 15.0D0 
         XU(5) = 15.0D0 
         FEX = 10.3D0
      GOTO 9999
C
C   MITP88
C
   88 CONTINUE
         PNAM = 'MITP88 (TLN6)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 42
         NBIN = 6
         N = NINT + NCONT + NBIN
         M = 36
         ME = 0
         DO I = NINT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         DO I = 1,NINT
            XL(I) = 0.0D0
            XU(I) = 5.0D0
            X(I) = 1.0D0
         ENDDO
         XU(1) = 16.0D0
         XU(2) = 16.0D0
         XU(3) = 16.0D0
         XU(4) = 16.0D0
         XU(5) = 16.0D0
         XU(6) = 16.0D0
         FEX = 0.146D+02 
      GOTO 9999
C
C   MITP89
C
   89 CONTINUE
         PNAM = 'MITP89 (PROB02)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 6
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 8
         ME = 0
         DO I = 1,N
            XL(I) = 1.0D0
            XU(I) = 100.0D0
            X(I) = 50.0D0
         ENDDO        
         FEX = 11.22350D4
      GOTO 9999
C
C   MITP90
C
   90 CONTINUE
         PNAM = 'MITP90 (TLOSS)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 42
         NBIN = 6
         N = NINT + NCONT + NBIN
         M = 53
         ME = 0
         DO I = 1,NINT
            XL(I) = 0.0D0
            XU(I) = 5.0D0
            X(I) = 1.0D0
         ENDDO
         DO I = NINT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         XU(37) = 15.0D0
         XU(38) = 12.0D0
         XU(39) = 8.0D0
         XU(40) = 7.0D0
         XU(41) = 4.0D0
         XU(42) = 2.0D0         
         FEX = 16.3D0
      GOTO 9999
C
C   MITP91
C
   91 CONTINUE
         PNAM = 'MITP91 (TLTR)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 36
         NBIN = 12
         N = NINT + NCONT + NBIN
         M = 54
         ME = 0
         DO I = 1,NINT
            XL(I) = 0.0D0
            XU(I) = 5.0D0
            X(I) = 1.0D0
         ENDDO
         DO I = NINT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         XU(28) = 100.0D0
         XU(29) = 100.0D0
         XU(30) = 100.0D0
         XU(31) = 100.0D0
         XU(32) = 100.0D0
         XU(33) = 100.0D0
         XU(34) = 100.0D0
         XU(35) = 100.0D0
         XU(36) = 100.0D0
         X(1)  = 1.0D0
         X(10) = 1.0D0
         X(20) = 1.0D0
         X(28) = 15.0D0
         X(29) = 80.0D0
         FEX = 48.0666666666667D0
      GOTO 9999      
C
C   MITP92
C
   92 CONTINUE
         PNAM = 'MITP92 (ALAN)'
         PREF = '_cite{MINLPLib}'
         NCONT = 4
         NINT = 0
         NBIN = 4
         N = NINT + NCONT + NBIN
         M = 7
         ME = 2
         DO I = 1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO
         FEX =  2.925D0
      GOTO 9999      
C
C   MITP93
C
   93 CONTINUE
         PNAM = 'MITP93 (MEANVARX)'
         PREF = '_cite{MINLPLib}'
         NCONT = 21
         NINT = 0
         NBIN = 14
         N = NINT + NCONT + NBIN
         M = 44
         ME = 8
         DO I = 1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO        
         XL(8) = 0.03D0
         XU(8) = 0.11D0 
         XL(9) = 0.04D0
         XU(9) = 0.1D0 
         XL(10) = 0.04D0
         XU(10) = 0.07D0
         XL(11) = 0.03D0
         XU(11) = 0.11D0 
         XL(12) = 0.03D0
         XU(12) = 0.2D0 
         XL(13) = 0.03D0
         XU(13) = 0.1D0 
         XL(14) = 0.03D0
         XU(14) = 0.1D0 
         XL(15) = 0.02D0
         XU(15) = 0.2D0 
         XL(16) = 0.02D0
         XU(16) = 0.15D0     
         XL(17) = 0.0D0
         XU(17) = 0.0D0 
         XL(18) = 0.0D0
         XU(18) = 0.0D0 
         XL(19) = 0.04D0
         XU(19) = 0.1D0 
         XL(20) = 0.04D0
         XU(20) = 0.15D0 
         XL(21) = 0.04D0
         XU(21) = 0.2D0 
         DO I = 8,21
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0*(XL(I)+XU(I))
         ENDDO                
         FEX = 14.18973D0
      GOTO 9999      
C
C   MITP94
C
   94 CONTINUE
         PNAM = 'MITP94 (HMITTELMANN)'
         PREF = '_cite{MINLPLib}'
         NCONT = 0
         NINT = 0
         NBIN = 16
         N = NINT + NCONT + NBIN
         M = 7
         ME = 0
         DO I = 1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 0.5D0
         ENDDO        
         FEX = 13.0D0
      GOTO 9999
C
C   MITP95
C
   95 CONTINUE
         PNAM = 'MITP95 (MIP-EX)'
         PREF = '_cite{GK97}'
         NCONT = 2
         NINT = 0
         NBIN = 3
         N = NINT + NCONT + NBIN
         M = 7
         ME = 0
         DO I = 1,NCONT
            XL(I) = 1.0D0
            XU(I) = 4.0D0
            X(I) = 1.0D0
         ENDDO        
         DO I = NCONT+1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = 1.0D0
         ENDDO        
         FEX = 3.5D0
      GOTO 9999
C
C   MITP96 
C
   96 CONTINUE
         PNAM = 'MITP96 (mgrid__cycles 1)'
         PREF = '_cite{TGKRO9}'
         M = 1
         ME = 0
         NBIN = 0
         NCONT = 0
         NINT = 5
         N = NINT + NCONT + NBIN
         XL(1) = 0.0D0
         XL(2) = 0.0D0
         XL(3) = 0.0D0    
         XL(4) = 0.0D0
         XL(5) = 0.0D0      
         XU(1) = 100.D0
         XU(2) = 100.D0
         XU(3) = 100.D0
         XU(4) = 100.D0
         XU(5) = 100.D0
         X(1) = 0.0D0
         X(2) = 0.0D0
         X(3) = 0.0D0
         X(4) = 0.0D0
         X(5) = 0.0D0
         FEX = 8.0D0
      GOTO 9999
C
C   MITP97
C
   97 CONTINUE
         PNAM = 'MITP97 (mgrid__cycles 2)'
         PREF = '_cite{TGKRO9}'
         M = 1
         ME = 0
         NBIN = 0
         NCONT = 0
         NINT = 10
         N = NINT + NCONT + NBIN
         XL(1) = 0.0D0
         XL(2) = 0.0D0
         XL(3) = 0.0D0    
         XL(4) = 0.0D0
         XL(5) = 0.0D0
         XL(6) = 0.0D0
         XL(7) = 0.0D0
         XL(8) = 0.0D0    
         XL(9) = 0.0D0
         XL(10) = 0.0D0 
         XU(1) = 100.D0
         XU(2) = 100.D0
         XU(3) = 100.D0
         XU(4) = 100.D0
         XU(5) = 100.D0
         XU(6) = 100.D0
         XU(7) = 100.D0
         XU(8) = 100.D0
         XU(9) = 100.D0
         XU(10) = 100.D0
         X(1) = 100.0D0
         X(2) = 100.0D0
         X(3) = 100.0D0
         X(4) = 100.0D0
         X(5) = 100.0D0
         X(6) = 100.0D0
         X(7) = 100.0D0
         X(8) = 100.0D0
         X(9) = 100.0D0
         X(10) = 100.0D0
         FEX = 300.0D0
      GOTO 9999 
C
C   MITP98
C
   98 CONTINUE
         PNAM = 'MITP98 (CROP20)'
         PREF = '\cite{SRL06}'
         NINT = 20
c         FEX = 0.13178478D+00
C         FEX = 0.68528060D+00
         FEX = 0.13185D0
         GOTO 9111
C
C   MITP99
C
   99 CONTINUE
         PNAM = 'MITP99 (CROP50)'
         PREF = '\cite{SRL06}'
         NINT = 50
c         FEX = 0.45824230D+00
C         FEX = 0.17132015D+01 
         FEX =  0.40524999D0
         GOTO 9111
C
C   MITP100
C
  100 CONTINUE
         PNAM = 'MITP100 (CROP100)'
         PREF = '\cite{SRL06}'
         NINT = 100
         FEX =  1.0973D0
         GOTO 9111
 9111    NCONT = 0
         NBIN = 0
         N = NINT
         M = 3
         ME = 0
         DO I = 1,N
            XL(I) = 1.0D0
            XU(I) = 5.0D0
            X(I) = 1.0D0
         ENDDO
         GU1 = 0.0D0
         GU2 = 0.0D0
         GU3 = 0.0D0
         GL1 = 0.0D0
         GL2 = 0.0D0
         GL3 = 0.0D0
         DO I=1,NINT
            R_CROP(I) = 0.8D0 + R_NUM(I)*0.18D0
            A1_CROP(I) = 1.0D0 + R_NUM(NINT+I)*49.0D0
            A2_CROP(I) = 1.0D0 + R_NUM(2*NINT+I)*49.0D0
            A3_CROP(I) = 1.0D0 + R_NUM(3*NINT+I)*49.0D0
            DELTA_CROP(I) = R_NUM(4*NINT+I)*0.01D0
            TAU_CROP(I) = R_NUM(5*NINT+I)*0.01D0
         ENDDO   
         DO I=1,NINT
            GU1 = GU1 + A1_CROP(I)*XU(I)**2
            GU2 = GU2 + A2_CROP(I)*(XU(I) + DEXP(DELTA_CROP(I)*XU(I)))
            GU3 = GU3 + A3_CROP(I)*XU(I)*DEXP(TAU_CROP(I)*XU(I))
            GL1 = GL1 + A1_CROP(I)*XL(I)**2
            GL2 = GL2 + A2_CROP(I)*(XL(I) + DEXP(DELTA_CROP(I)*XL(I)))
            GL3 = GL3 + A3_CROP(I)*XL(I)*DEXP(TAU_CROP(I)*XL(I))
         ENDDO
         B1_CROP = GL1 + 0.3D0*(GU1 - GL1)
         B2_CROP = GL2 + 0.3D0*(GU2 - GL2)
         B3_CROP = GL3 + 0.3D0*(GU3 - GL3)
      GOTO 9999 
C
C   MITP101 (KNAPSACK)
C
  101 CONTINUE
         PNAM = 'KNAPSACK'
         PREF = ''
         NCONT = 0
         NINT  = 0
         NBIN  = 50
         M     = 1
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = 0.0D0
         XU(I) = 1.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = -1.55421218D0
         GOTO 9999
C
C   MITP102 (MINLP MIX)
C
  102 CONTINUE
         PNAM = 'MINLP Mix'
         PREF = ''
         NCONT = 10
         NINT  = 10
         NBIN  = 0
         M     = 3
         ME    = 2
         N = NINT + NCONT + NBIN
         DO I = 1,2 ! Rastringin
             XL(I) = -1.0D0
             XU(I) =  6.0D0
         ENDDO
         DO I = 3,4 ! Griewank
             XL(I) = -600.0D0
             XU(I) =  700.0D0
         ENDDO  
         DO I = 5,6 ! Schwefel
             XL(I) = -500.0D0
             XU(I) =  500.0D0
         ENDDO
         DO I = 7,8 ! Ackley
             XL(I) = -90.0D0
             XU(I) =  80.0D0
         ENDDO       
         DO I = 9,10 ! Rosenbrock
             XL(I) = -5.0D0
             XU(I) =  4.0D0
         ENDDO    
         DO I = 11,20 ! Integer
             XL(I) = -10.0D0
             XU(I) =  1.0D0 + DBLE(I)       
         ENDDO                          
         FEX = 0.0001D0
         GOTO 9999
C
C   MITP103 (Polynom)
C
  103 CONTINUE
         PNAM = 'Polynom'
         PREF = ''
         NCONT = 2
         NINT  = 1
         NBIN  = 0
         M     = 3
         ME    = 3
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = 0.0D0
         XU(I) = 100.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D0+2.0D0+3.0D0
         GOTO 9999 
C
C   MITP104 (Polynom)
C
  104 CONTINUE
         PNAM = 'Polynom'
         PREF = ''
         NCONT = 2
         NINT  = 3
         NBIN  = 0
         M     = 5
         ME    = 5
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = 0.0D0
         XU(I) = 100.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D0+2.0D0+3.0D0+4.0D0+5.0D0
         GOTO 9999 
C
C   MITP105 (Polynom)
C
  105 CONTINUE
         PNAM = 'Polynom'
         PREF = ''
         NCONT = 3
         NINT  = 2
         NBIN  = 0
         M     = 5
         ME    = 5
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = 0.0D0
         XU(I) = 100.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D0+2.0D0+3.0D0+4.0D0+5.0D0
         GOTO 9999
C
C   MITP106 (Polynom)
C
  106 CONTINUE
         PNAM = 'Polynom'
         PREF = ''
         NCONT = 1
         NINT  = 6
         NBIN  = 0
         M     = 7
         ME    = 7
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = 0.0D0
         XU(I) = 100.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D0+2.0D0+3.0D0+4.0D0+5.0D0+6.0D0+7.0D0
         GOTO 9999
C
C   MITP107 (Polynom)
C
  107 CONTINUE
         PNAM = 'Polynom'
         PREF = ''
         NCONT = 1
         NINT  = 8
         NBIN  = 0
         M     = 9
         ME    = 9
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = 0.0D0
         XU(I) = 100.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D0+2.0D0+3.0D0+4.0D0+5.0D0+6.0D0+7.0D0+8.0D0+9.0D0
         GOTO 9999              
C
C   MITP108 Shell WellRelinking333_600(small)
C
  108 CONTINUE
      PNAM= 'MITP108(SHELL WR333__600)'
      M=9
      ME=3
      NBIN=9
      NCONT=3
      NINT=0
      N = NINT + NCONT + NBIN
      
C     BOUNDS FOR Q1, Q2 AND Q3      
      
       XL(1)= 0.0D0
       XL(2)= 0.0D0
       XL(3)= 0.0D0
      
       XU(1)= 900.D0
       XU(2)= 800.D0
       XU(3)= 700.D0
      
C     BOUNDS FOR SW1_SF1, ... SW3_SF3:
       XL(4)= 0.0D0
       XL(5)= 0.0D0
       XL(6)= 0.0D0
       XL(7)= 0.0D0
       XL(8)= 0.0D0
       XL(9)= 0.0D0
       XL(10)= 0.0D0
       XL(11)= 0.0D0
       XL(12)= 0.0D0
      
      XU(4)= 1.0D0
      XU(5)= 1.0D0
      XU(6)= 1.0D0
      XU(7)= 1.0D0
      XU(8)= 1.0D0
      XU(9)= 1.0D0
      XU(10)= 1.0D0
      XU(11)= 1.0D0
      XU(12)= 1.0D0

       FEX= -1604.5D0
       FEX=(1.0D+4+FEX)*FSCALE

      GOTO 9999                        
C
C   MITP109 Shell WellRelinking663_600(medium)
C
  109 CONTINUE
      PNAM= 'MITP109(SHELL WR663__600)'
      M=15
      ME=6
      NBIN=18
      NCONT=6
      NINT=0
      N = NINT + NCONT + NBIN
      
C     BOUNDS FOR Q1, Q2 AND Q3      
      
       XL(1)= 0.0D0
       XL(2)= 0.0D0
       XL(3)= 0.0D0    
       XL(4)= 0.0D0
       XL(5)= 0.0D0
       XL(6)= 0.0D0
       
       XU(1)= 900.D0
       XU(2)= 800.D0
       XU(3)= 700.D0
       XU(4)= 600.D0
       XU(5)= 500.D0
       XU(6)= 400.D0
       
C     BOUNDS FOR SW1_SF1, ... SW3_SF3:
       XL(7)= 0.0D0
       XL(8)= 0.0D0
       XL(9)= 0.0D0
       XL(10)= 0.0D0
       XL(11)= 0.0D0
       XL(12)= 0.0D0
       XL(13)= 0.0D0
       XL(14)= 0.0D0
       XL(15)= 0.0D0
       XL(16)= 0.0D0
       XL(17)= 0.0D0
       XL(18)= 0.0D0
       XL(19)= 0.0D0
       XL(20)= 0.0D0
       XL(21)= 0.0D0
       XL(22)= 0.0D0
       XL(23)= 0.0D0
       XL(24)= 0.0D0
      
       XU(7)= 1.0D0
       XU(8)= 1.0D0
       XU(9)= 1.0D0
       XU(10)= 1.0D0
       XU(11)= 1.0D0
       XU(12)= 1.0D0
       XU(13)= 1.0D0
       XU(14)= 1.0D0
       XU(15)= 1.0D0
       XU(16)= 1.0D0
       XU(17)= 1.0D0
       XU(18)= 1.0D0
       XU(19)= 1.0D0
       XU(20)= 1.0D0
       XU(21)= 1.0D0
       XU(22)= 1.0D0
       XU(23)= 1.0D0
       XU(24)= 1.0D0
             
       FEX= -1800.D0
       FEX=(1.0D+4+FEX)*FSCALE

      GOTO 9999                        
C
C   MITP110 Shell WellRelinking663_800(medium)
C
  110 CONTINUE
      PNAM= 'MITP110(SHELL WR663__800)'
      M=15
      ME=6
      NBIN=18
      NCONT=6
      NINT=0
      N = NINT + NCONT + NBIN
      
C     BOUNDS FOR Q1, Q2 AND Q3      
      
       XL(1)= 0.0D0
       XL(2)= 0.0D0
       XL(3)= 0.0D0    
       XL(4)= 0.0D0
       XL(5)= 0.0D0
       XL(6)= 0.0D0
       
       XU(1)= 900.D0
       XU(2)= 800.D0
       XU(3)= 700.D0
       XU(4)= 600.D0
       XU(5)= 500.D0
       XU(6)= 400.D0
       
C     BOUNDS FOR SW1_SF1, ... SW3_SF3:
       XL(7)= 0.0D0
       XL(8)= 0.0D0
       XL(9)= 0.0D0
       XL(10)= 0.0D0
       XL(11)= 0.0D0
       XL(12)= 0.0D0
       XL(13)= 0.0D0
       XL(14)= 0.0D0
       XL(15)= 0.0D0
       XL(16)= 0.0D0
       XL(17)= 0.0D0
       XL(18)= 0.0D0
       XL(19)= 0.0D0
       XL(20)= 0.0D0
       XL(21)= 0.0D0
       XL(22)= 0.0D0
       XL(23)= 0.0D0
       XL(24)= 0.0D0
      
       XU(7)= 1.0D0
       XU(8)= 1.0D0
       XU(9)= 1.0D0
       XU(10)= 1.0D0
       XU(11)= 1.0D0
       XU(12)= 1.0D0
       XU(13)= 1.0D0
       XU(14)= 1.0D0
       XU(15)= 1.0D0
       XU(16)= 1.0D0
       XU(17)= 1.0D0
       XU(18)= 1.0D0
       XU(19)= 1.0D0
       XU(20)= 1.0D0
       XU(21)= 1.0D0
       XU(22)= 1.0D0
       XU(23)= 1.0D0
       XU(24)= 1.0D0
              
       FEX= -2308.3D0
C       FEX=(1.0D+4+FEX)*FSCALE

      GOTO 9999                        

C
C   MITP111 Shell WellRelinking663_900(medium)
C
  111 CONTINUE
      PNAM= 'MITP111(SHELL WR663__900)'
      M=15
      ME=6
      NBIN=18
      NCONT=6
      NINT=0
      N = NINT + NCONT + NBIN
      
C     BOUNDS FOR Q1, Q2 AND Q3      
      
       XL(1)= 0.0D0
       XL(2)= 0.0D0
       XL(3)= 0.0D0    
       XL(4)= 0.0D0
       XL(5)= 0.0D0
       XL(6)= 0.0D0
       
       XU(1)= 900.D0
       XU(2)= 800.D0
       XU(3)= 700.D0
       XU(4)= 600.D0
       XU(5)= 500.D0
       XU(6)= 400.D0
       
C     BOUNDS FOR SW1_SF1, ... SW3_SF3:
       XL(7)= 0.0D0
       XL(8)= 0.0D0
       XL(9)= 0.0D0
       XL(10)= 0.0D0
       XL(11)= 0.0D0
       XL(12)= 0.0D0
       XL(13)= 0.0D0
       XL(14)= 0.0D0
       XL(15)= 0.0D0
       XL(16)= 0.0D0
       XL(17)= 0.0D0
       XL(18)= 0.0D0
       XL(19)= 0.0D0
       XL(20)= 0.0D0
       XL(21)= 0.0D0
       XL(22)= 0.0D0
       XL(23)= 0.0D0
       XL(24)= 0.0D0
      
       XU(7)= 1.0D0
       XU(8)= 1.0D0
       XU(9)= 1.0D0
       XU(10)= 1.0D0
       XU(11)= 1.0D0
       XU(12)= 1.0D0
       XU(13)= 1.0D0
       XU(14)= 1.0D0
       XU(15)= 1.0D0
       XU(16)= 1.0D0
       XU(17)= 1.0D0
       XU(18)= 1.0D0
       XU(19)= 1.0D0
       XU(20)= 1.0D0
       XU(21)= 1.0D0
       XU(22)= 1.0D0
       XU(23)= 1.0D0
       XU(24)= 1.0D0
              
       FEX= -2508.3D0
       FEX=(1.0D+4+FEX)*FSCALE

      GOTO 9999  

C
C   MITP112 
C
  112 CONTINUE
          PNAM= 'GENO EXAMPLE 1'
          NCONT =3      
          NINT  =4
          NBIN  =0  
          N = NINT + NCONT + NBIN      
          M  = 9
          ME = 0         
          DO I = 1,NCONT
              XL(I) = 0.0D0
              XU(I) = 100.0D0
          ENDDO
          DO I = NCONT+1,N
              XL(I) = 0.0D0
              XU(I) = 1.0D0    
          ENDDO
          X(1) = 0.138449d0
          X(2) = 0.799999d0
          X(3) = 2.061552d0
          X(4) = 1.0d0
          X(5) = 1.0d0
          X(6) = 0.0d0
          X(7) = 1.0d0
C          GENO Loesung          
C          FEX = 3.36981D0
C         MIDACO Loesung
          FEX = 3.557466D0
      GOTO 9999  
C
C   MITP113 
C
  113 CONTINUE
         PNAM= 'GENO EXAMPLE 2'
          NCONT =2      
          NINT  =1
          NBIN  =0  
          N = NINT + NCONT + NBIN      
          M  = 4
          ME = 0         
          DO I = 1,NCONT
              XL(I) = 1.0D-12
              XU(I) = 100.0D0
          ENDDO
          DO I = NCONT+1,N
              XL(I) = 0.0D0
              XU(I) = 1.0D0    
          ENDDO
          X(1) = 3.514237d0
          X(2) = 1.0d-12
          X(3) = 1.0d0
C          GENO Loesung          
          FEX = 99.239635D0
      GOTO 9999  

C
C   MITP114  (Altes PLATO 100 Beispiel)
C
  114 CONTINUE
      PNAM= 'PLATO 100'
      M=9
      ME=4
      NBIN=0
      NCONT=3
      NINT=4
      N = NINT + NCONT + NBIN
      DO I = 1,N
          XL(I) = -10.0D0
          XU(I) =  10.0D0
      ENDDO
      XL(6) = 0.0D0
      XL(7) = 0.0D0
      XU(6) = 1.0D0
      XU(7) = 1.0D0
      
      X(1) = 6.0D0
      X(2) = 5.0D0
      X(3) = 4.0D0
      X(4) = 3.0D0
      X(5) = 2.0D0
      X(6) = 1.0D0
      X(7) = 0.0D0
      
      FEX= 1.0D-5
      GOTO 9999  
 
C
C   MITP115 (60er BINARY)
C
  115 CONTINUE
          PNAM  = '60er BINARY'
          NCONT = 0
          NINT  = 0
          NBIN  = 60
          N = NINT + NCONT + NBIN   
          M     = 20
          ME    = 20                 
          DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 1.0D0
          ENDDO   
          FEX = -40.0D0
      GOTO 9999    
C
C   MITP116 (120er BINARY)
C
  116 CONTINUE
          PNAM  = '120er BINARY'
          NCONT = 0
          NINT  = 0
          NBIN  = 120
          N = NINT + NCONT + NBIN   
          M     = 60
          ME    = 30                
          DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 1.0D0
          ENDDO   
          FEX = -120.0D0
      GOTO 9999  
C
C   MITP117 Shell WellRelinking993_1500(LARGE)
C
  117 CONTINUE
      PNAM= 'MITP117(SHELL WR993__1500)'
      M=21
      ME=9
      NBIN=27
      NCONT=9
      NINT=0
      N = NINT + NCONT + NBIN
 
C     BOUNDS FOR Q1, Q2 AND Q3      
      
       XL(1)= 0.0D0
       XL(2)= 0.0D0
       XL(3)= 0.0D0    
       XL(4)= 0.0D0
       XL(5)= 0.0D0
       XL(6)= 0.0D0
       XL(7)= 0.0D0
       XL(8)= 0.0D0
       XL(9)= 0.0D0  
       
       XU(1)= 900.D0
       XU(2)= 800.D0
       XU(3)= 700.D0
       XU(4)= 600.D0
       XU(5)= 500.D0
       XU(6)= 400.D0
       XU(7)= 300.0D0
       XU(8)= 200.0D0
       XU(9)= 100.0D0
       
C     BOUNDS FOR SW1_SF1, ... SW3_SF3:
       
       XL(10)= 0.0D0
       XL(11)= 0.0D0
       XL(12)= 0.0D0
       XL(13)= 0.0D0
       XL(14)= 0.0D0
       XL(15)= 0.0D0
       XL(16)= 0.0D0
       XL(17)= 0.0D0
       XL(18)= 0.0D0
       XL(19)= 0.0D0
       XL(20)= 0.0D0
       XL(21)= 0.0D0
       XL(22)= 0.0D0
       XL(23)= 0.0D0
       XL(24)= 0.0D0
       XL(25)= 0.0D0
       XL(26)= 0.0D0
       XL(27)= 0.0D0
       XL(28)= 0.0D0
       XL(29)= 0.0D0
       XL(30)= 0.0D0
       XL(31)= 0.0D0
       XL(32)= 0.0D0
       XL(33)= 0.0D0
       XL(34)= 0.0D0
       XL(35)= 0.0D0
       XL(36)= 0.0D0

       
       XU(10)= 1.0D0
       XU(11)= 1.0D0
       XU(12)= 1.0D0
       XU(13)= 1.0D0
       XU(14)= 1.0D0
       XU(15)= 1.0D0
       XU(16)= 1.0D0
       XU(17)= 1.0D0
       XU(18)= 1.0D0
       XU(19)= 1.0D0
       XU(20)= 1.0D0
       XU(21)= 1.0D0
       XU(22)= 1.0D0
       XU(23)= 1.0D0
       XU(24)= 1.0D0
       XU(25)= 1.0D0
       XU(26)= 1.0D0
       XU(27)= 1.0D0
       XU(28)= 1.0D0
       XU(29)= 1.0D0
       XU(30)= 1.0D0
       XU(31)= 1.0D0
       XU(32)= 1.0D0
       XU(33)= 1.0D0
       XU(34)= 1.0D0
       XU(35)= 1.0D0
       XU(36)= 1.0D0

       FEX= -3404.5D0
       FEX=(1.0D4+FEX)*FSCALE

      GOTO 9999 
C
C   MITP118 Testcase Erlangen (Alexander Thekale) small 1 
C
  118 CONTINUE
      PNAM= 'MITP118(Erlangen)'
      M=1
      ME=0
      NBIN=0
      NCONT=0
      NINT=5
      N = NINT + NCONT + NBIN
 
C     BOUNDS    
      
       XL(1)= 0.0D0
       XL(2)= 0.0D0
       XL(3)= 0.0D0    
       XL(4)= 0.0D0
       XL(5)= 0.0D0
       
       XU(1)= 100.D0
       XU(2)= 100.D0
       XU(3)= 100.D0
       XU(4)= 100.D0
       XU(5)= 100.D0
       
C     STARTING VALUES      
       X0(1)= 0.0D0
       X0(2)= 0.0D0
       X0(3)= 0.0D0
       X0(4)= 0.0D0
       X0(5)= 0.0D0
 
       XEX(1)= 2.0D0
       XEX(2)= 1.0D0
       XEX(3)= 0.0D0
       XEX(4)= 0.0D0
       XEX(5)= 0.0D0
       
       FEX= 8.0D0*0.1D0
      GOTO 9999
C
C   MITP119 Testcase Erlangen (Alexander Thekale) large 1 
C
  119 CONTINUE
      PNAM= 'MITP119(Erlangen)'
      M=1
      ME=0
      NBIN=0
      NCONT=0
      NINT=10
      N = NINT + NCONT + NBIN
 
C     BOUNDS    
      
       XL(1)= 0.0D0
       XL(2)= 0.0D0
       XL(3)= 0.0D0    
       XL(4)= 0.0D0
       XL(5)= 0.0D0
       XL(6)= 0.0D0
       XL(7)= 0.0D0
       XL(8)= 0.0D0    
       XL(9)= 0.0D0
       XL(10)= 0.0D0
       
       XU(1)= 100.D0
       XU(2)= 100.D0
       XU(3)= 100.D0
       XU(4)= 100.D0
       XU(5)= 100.D0
       XU(6)= 100.D0
       XU(7)= 100.D0
       XU(8)= 100.D0
       XU(9)= 100.D0
       XU(10)= 100.D0
              
       FEX= 300.0D0*0.1D0

      GOTO 9999 
C
C   MITP120 'Rosenbrock + Integer'
C
  120 CONTINUE
         PNAM = 'Rosenbrock + Integer'
         NCONT = 10
         NINT = 2
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 2
         ME = 2
          DO I = 1,N
             XL(I) = -10.0D0
             XU(I) =  50.0D0
             X(I) = 0.0D0
         ENDDO 
         FEX = 0.0001D0
      GOTO 9999      
C
C   MITP121 'LARGE MI-LP'
C
  121 CONTINUE
         PNAM = 'LARGE MI-LP'
         NCONT = 30
         NINT = 150
         NBIN = 0
         N = NINT + NCONT + NBIN
         M = 0
         ME = 0
          DO I = 1,NCONT
             XL(I) = -50.0D0
             XU(I) =  1.0D0
             X(I) = 0.0D0
         ENDDO
         XL(2)  = -1000.0D0
         XL(14) = -7000000.0D0 
         DO I = 1,NINT
             XL(NCONT+I) =  0.0D0
             XU(NCONT+I) =  1.0D0
             X(I) = 0.0D0
         ENDDO  
         XL(190) = -2000000.0D0

         FEX = -180.0D0
      GOTO 9999
      
C
C   MITP122 'BIG MINLP'
C
  122 CONTINUE
         PNAM = 'MITP122 BIG MINLP'
         NCONT = 6
         NINT = 50
         NBIN = 0
         N  = NINT + NCONT + NBIN
         M  = 0
         ME = 0
         DO I = 1,NCONT
             XL(I) = -5.12D0
             XU(I) =  5.24D0
             X(I) = 0.0D0
         ENDDO
         DO I = 1,NINT
             XL(NCONT+I) = - 1.0D0 - DNINT(DBLE(I)/10.0D0)
             XU(NCONT+I) =   1.0D0
             X(I) = 0.0D0
         ENDDO         

         FEX = -50.0D0
      GOTO 9999        

C
C   MITP123 (HORROR CHAOS)
  123 CONTINUE
          PNAM  = 'HORROR CHAOS'
          NCONT = 2
          NINT  = 2
          NBIN  = 0
          M     = 4
          ME    = 2
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = -10.0D0
             XU(I) =  50.0D0
             X(I) = 0.0D0
          ENDDO   
          FEX = 123.0D0
      GOTO 9999 
      
      
C
C   MITP124 Shell Gas Lift Version 0.3
C     
  124 CONTINUE
      PNAM= 'MITP124(SHELL GasLift3)'
      M=9
      ME=0
      NBIN=4
      NCONT=8
      NINT=0
      N = NINT + NCONT + NBIN
 
C     BOUNDS FOR c_lgi      
      LGTOT = 0.3D3
      
       XL(1)= 0.0D0
       XL(2)= 0.0D0
       XL(3)= 0.0D0    
       XL(4)= 0.0D0
       
       XU(1)= DMIN1(LGTOT,3000.0D0)
       XU(2)= DMIN1(LGTOT,3000.0D0)
       XU(3)= DMIN1(LGTOT,3000.0D0)
       XU(4)= DMIN1(LGTOT,3000.0D0)
       
C     BOUNDS FOR THPiss       
       XL(5)= 250.0D0
       XL(6)= 250.0D0
       XL(7)= 250.0D0
       XL(8)= 250.0D0
              
       XU(5)= 1500.D0
       XU(6)= 1400.D0
       XU(7)= 1000.D0
       XU(8)= 900.D0
       
C     BOUNDS FOR BINARIES:
       XL(9)= 0.0D0
       XL(10)= 0.0D0
       XL(11)= 0.0D0
       XL(12)= 0.0D0
       
       XU(9)= 1.0D0
       XU(10)= 1.0D0
       XU(11)= 1.0D0
       XU(12)= 1.0D0
       
       FEX= -4.3873D3
       FEX = FEX*SCGL

      GOTO 9999                          
C
C   MITP125 Shell Gas Lift Version 0.3
C     
  125 CONTINUE
      PNAM= 'MITP125(SHELL GasLift4)'
      M=9
      ME=0
      NBIN=4
      NCONT=8
      NINT=0
      N = NINT + NCONT + NBIN
 
C     BOUNDS FOR c_lgi      
      LGTOT = 0.4D3
      
       XL(1)= 0.0D0
       XL(2)= 0.0D0
       XL(3)= 0.0D0    
       XL(4)= 0.0D0
       
       XU(1)= DMIN1(LGTOT,3000.D0)
       XU(2)= DMIN1(LGTOT,3000.D0)
       XU(3)= DMIN1(LGTOT,3000.D0)
       XU(4)= DMIN1(LGTOT,3000.D0)
       
C     BOUNDS FOR THPiss       
       XL(5)= 250.0D0
       XL(6)= 250.0D0
       XL(7)= 250.0D0
       XL(8)= 250.0D0
              
       XU(5)= 1500.D0
       XU(6)= 1400.D0
       XU(7)= 1000.D0
       XU(8)= 900.D0
       
       
       
         
C     BOUNDS FOR BINARIES:
       XL(9)= 0.0D0
       XL(10)= 0.0D0
       XL(11)= 0.0D0
       XL(12)= 0.0D0
       
       XU(9)= 1.0D0
       XU(10)= 1.0D0
       XU(11)= 1.0D0
       XU(12)= 1.0D0

       FEX= -4.6699D3
       FEX = FEX*SCGL
      GOTO 9999     
C
C   MITP126 (Stairs)
C
  126 CONTINUE
         PNAM = 'Stairs'
         PREF = ''
         NCONT = 5
         NINT  = 5
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = -10.0D0
         XU(I) = 100000.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-5
       GOTO 9999
C
C   MITP127 (Stairs)
C
  127 CONTINUE
         PNAM = 'Stairs'
         PREF = ''
         NCONT = 15
         NINT  = 5
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = -1000.0D0
         XU(I) = 1000.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-5
       GOTO 9999    
C
C   MITP128 (Stairs)
C
  128 CONTINUE
         PNAM = 'Stairs'
         PREF = ''
         NCONT = 10
         NINT  = 20
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = -1000.0D0
         XU(I) = 700.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-5
       GOTO 9999
C
C   MITP129 (Stairs)
C
  129 CONTINUE
         PNAM = 'Stairs'
         PREF = ''
         NCONT = 5
         NINT  = 30
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = -500.0D0
         XU(I) = 100.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-5
       GOTO 9999
C
C   MITP130 (64 binary)
C
  130 CONTINUE
         PNAM = '64 binary'
         PREF = ''
         NCONT = 0
         NINT  = 0
         NBIN  = 64
         N = NINT + NCONT + NBIN         
         M     = N/4
         ME    = N/4
         DO I = 1,N
         XL(I) = 0.0D0
         XU(I) = 1.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = N/4
       GOTO 9999
C
C   MITP131 (EQ-CONSTRAINTS)
C
  131 CONTINUE
         PNAM = 'EQ-CONSTRAINTS'
         PREF = ''
         NCONT = 2
         NINT  = 2
         NBIN  = 0
         M     = 3
         ME    = 3
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -5.0D0
c             XU(I) = 5000.0D0
             XU(I) = 5.0D0             
              X(I) = XL(I) 
         ENDDO
         FEX = 1.0D0
       GOTO 9999
C
C   MITP132 
C
  132 CONTINUE
         PNAM = ''
         PREF = ''
         NCONT = 5
         NINT  = 5
         NBIN  = 0
         M     = 5
         ME    = 5
         N = NINT + NCONT + NBIN
         DO I = 1,5
             XL(I) = -30.0D0
             XU(I) = 10.0D0
              X(I) = XL(I) 
         ENDDO
         DO I = 6,10
             XL(I) = -5.0D0
             XU(I) = 7.0D0
              X(I) = XL(I) 
         ENDDO
         FEX = 0.001D0
       GOTO 9999
C
C   MITP133 (EQ-CONSTRAINTS)
C
  133 CONTINUE
         PNAM = 'EQ-CONSTRAINTS'
         PREF = ''
         NCONT = 4
         NINT  = 5
         NBIN  = 0
         M     = 13
         ME    = 4
         N = NINT + NCONT + NBIN
         DO I = 1,NCONT
             XL(I) = -5.0D0
             XU(I) =  5.0D0
              X(I) = XL(I) 
         ENDDO
         DO I = NCONT+1,N
             XL(I) = 0.0D0
             XU(I) = 1.0D0
              X(I) = XL(I) 
         ENDDO         
         FEX = 1.0D0
       GOTO 9999
C
C   MITP134 (LP-UNCONSTRAINED)
C
  134 CONTINUE
         PNAM = 'LP-UNCONSTRAINED'
         NCONT = 30
         NINT  = 10
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,NCONT
             XL(I) = 0.0D0
             XU(I) = 100.0D0
              X(I) = XL(I) 
         ENDDO
         DO I = NCONT+1,N
             XL(I) = 0.0D0
             XU(I) = 5.0D0
              X(I) = XL(I) 
         ENDDO         
         FEX = - NCONT * XU(1) - NINT * XU(NCONT+1)
       GOTO 9999
C
C   MITP135 (LP-CONSTRAINED)
C
  135 CONTINUE
         PNAM = 'LP-CONSTRAINED'
         NCONT = 20
         NINT  = 10
         NBIN  = 0
         N = NINT + NCONT + NBIN         
         M     = N
         ME    = NINT
         DO I = 1,NCONT
             XL(I) = 0.0D0
             XU(I) = 10.0D0
              X(I) = XL(I) 
         ENDDO
         DO I = NCONT+1,N
             XL(I) = 0.0D0
             XU(I) = 6.0D0
              X(I) = XL(I) 
         ENDDO         
         FEX = - NCONT * XU(1) - NINT * 5.0D0
       GOTO 9999
C
C   MITP136 (BIG-BINARY)
C
  136 CONTINUE
         PNAM = 'BIG-BINARY'
         NCONT = 0
         NINT  = 200
         NBIN  = 0
         N = NINT + NCONT + NBIN         
         M     = 0
         ME    = 0
         DO I = 1,NINT
             XL(I) = 0.0D0
             XU(I) = 1.0D0
              X(I) = XL(I) 
         ENDDO       
         
         XL(10)  = -10.0D0
         XL(100) = -100.0D0         
         XL(200) = -1000.0D0         
         XL(300) = -5000.0D0 
         XL(400) = -50.0D0                  
           
         FEX = - N
       GOTO 9999
C
C   MITP137 (BIG-LP)
C
  137 CONTINUE
         PNAM = 'BIG-LP'
         NCONT = 140
         NINT  = 20
         NBIN  = 0
         N = NINT + NCONT + NBIN         
         M     = 0
         ME    = 0
         DO I = 1,NCONT
             XL(I) = 0.0D0
             XU(I) = 1.0D0
              X(I) = XL(I) 
         ENDDO
         DO I = NCONT+1,N
             XL(I) = 0.0D0
             XU(I) = 1.0D0
              X(I) = XL(I) 
         ENDDO         
         
         XL(5) = -50.0D0
         XL(50) = -500.0D0
         XL(90) = -9000.0D0         
         
         FEX = - N
       GOTO 9999
C
C   MITP138 (Precision LP)
C
  138 CONTINUE
         PNAM = 'Precision LP'
         NCONT = 50
         NINT  = 30
         NBIN  = 0
         N = NINT + NCONT + NBIN         
         M     = 0
         ME    = 0
         DO I = 1,NCONT
             XL(I) = 0.0D0
             XU(I) = 1.0D0
              X(I) = XL(I) 
         ENDDO
         DO I = NCONT+1,N
             XL(I) = 0.0D0
             XU(I) = 1.0D0
              X(I) = XL(I) 
         ENDDO         
         FEX = 0.0D0
          DO I = 1,NCONT
              FEX = FEX - XU(I) / I**2
          ENDDO 
          DO I = 1,NINT
              FEX = FEX - XU(NCONT+I) / I**2
          ENDDO         
       GOTO 9999           
C
C   MITP139 (LP&QP)
C
  139 CONTINUE
         PNAM = 'LP&QP'
         NCONT = 30
         NINT  = 10
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,NCONT
             XL(I) = -5.0D0
             XU(I) = 10.0D0
              X(I) = XL(I) 
         ENDDO
         DO I = NCONT+1,N
             XL(I) = 0.0D0
             XU(I) = 5.0D0
              X(I) = XL(I) 
         ENDDO         
         FEX = - NINT * XU(NCONT+1)
       GOTO 9999
C
C   MITP140 (LP&QP)
C
  140 CONTINUE
         PNAM = 'LP&QP'
         NCONT = 30
         NINT  = 40
         NBIN  = 0
         M     = 10
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,NCONT
             XL(I) = -1.0D0
             XU(I) =  2.0D0
              X(I) = XL(I) 
         ENDDO
         DO I = NCONT+1,N
             XL(I) = 0.0D0
             XU(I) = 5.0D0
              X(I) = XL(I) 
         ENDDO         
         FEX = - NINT * XU(NCONT+1)
       GOTO 9999   
C
C   MITP141 Shell WellRelinking663_1000(medium)
C
  141 CONTINUE
      PNAM= 'MITP112(SHELL WR663__1000)'
      M=15
      ME=6
      NBIN=18
      NCONT=6
      NINT=0
      N = NINT + NCONT + NBIN
      
C     BOUNDS FOR Q1, Q2 AND Q3      
      
       XL(1)= 0.0D0
       XL(2)= 0.0D0
       XL(3)= 0.0D0    
       XL(4)= 0.0D0
       XL(5)= 0.0D0
       XL(6)= 0.0D0
       
       XU(1)= 900.D0
       XU(2)= 800.D0
       XU(3)= 700.D0
       XU(4)= 600.D0
       XU(5)= 500.D0
       XU(6)= 400.D0
       
C     BOUNDS FOR SW1_SF1, ... SW3_SF3:
       XL(7)= 0.0D0
       XL(8)= 0.0D0
       XL(9)= 0.0D0
       XL(10)= 0.0D0
       XL(11)= 0.0D0
       XL(12)= 0.0D0
       XL(13)= 0.0D0
       XL(14)= 0.0D0
       XL(15)= 0.0D0
       XL(16)= 0.0D0
       XL(17)= 0.0D0
       XL(18)= 0.0D0
       XL(19)= 0.0D0
       XL(20)= 0.0D0
       XL(21)= 0.0D0
       XL(22)= 0.0D0
       XL(23)= 0.0D0
       XL(24)= 0.0D0
      
       XU(7)= 1.0D0
       XU(8)= 1.0D0
       XU(9)= 1.0D0
       XU(10)= 1.0D0
       XU(11)= 1.0D0
       XU(12)= 1.0D0
       XU(13)= 1.0D0
       XU(14)= 1.0D0
       XU(15)= 1.0D0
       XU(16)= 1.0D0
       XU(17)= 1.0D0
       XU(18)= 1.0D0
       XU(19)= 1.0D0
       XU(20)= 1.0D0
       XU(21)= 1.0D0
       XU(22)= 1.0D0
       XU(23)= 1.0D0
       XU(24)= 1.0D0

       FEX= -2626.6D0
       FEX=(1.0D+4+FEX)*FSCALE
      GOTO 9999  
C
C   MITP142 Shell WellRelinking663_1200(medium)
C
  142 CONTINUE
      PNAM= 'MITP113(SHELL WR663__1200)'
      M=15
      ME=6
      NBIN=18
      NCONT=6
      NINT=0
      N = NINT + NCONT + NBIN
      
C     BOUNDS FOR Q1, Q2 AND Q3      
      
       XL(1)= 0.0D0
       XL(2)= 0.0D0
       XL(3)= 0.0D0    
       XL(4)= 0.0D0
       XL(5)= 0.0D0
       XL(6)= 0.0D0
       
       XU(1)= 900.D0
       XU(2)= 800.D0
       XU(3)= 700.D0
       XU(4)= 600.D0
       XU(5)= 500.D0
       XU(6)= 400.D0
       
C     BOUNDS FOR SW1_SF1, ... SW3_SF3:
       XL(7)= 0.0D0
       XL(8)= 0.0D0
       XL(9)= 0.0D0
       XL(10)= 0.0D0
       XL(11)= 0.0D0
       XL(12)= 0.0D0
       XL(13)= 0.0D0
       XL(14)= 0.0D0
       XL(15)= 0.0D0
       XL(16)= 0.0D0
       XL(17)= 0.0D0
       XL(18)= 0.0D0
       XL(19)= 0.0D0
       XL(20)= 0.0D0
       XL(21)= 0.0D0
       XL(22)= 0.0D0
       XL(23)= 0.0D0
       XL(24)= 0.0D0
      
       XU(7)= 1.0D0
       XU(8)= 1.0D0
       XU(9)= 1.0D0
       XU(10)= 1.0D0
       XU(11)= 1.0D0
       XU(12)= 1.0D0
       XU(13)= 1.0D0
       XU(14)= 1.0D0
       XU(15)= 1.0D0
       XU(16)= 1.0D0
       XU(17)= 1.0D0
       XU(18)= 1.0D0
       XU(19)= 1.0D0
       XU(20)= 1.0D0
       XU(21)= 1.0D0
       XU(22)= 1.0D0
       XU(23)= 1.0D0
       XU(24)= 1.0D0

       FEX= -2804.5D0
       FEX=(1.0D+4+FEX)*FSCALE
      GOTO 9999  

C
C   MITP143 Shell WellRelinking663_1500(medium)
C
  143 CONTINUE
      PNAM= 'MITP114(SHELL WR663__1500)'
      M=15
      ME=6
      NBIN=18
      NCONT=6
      NINT=0
      N = NINT + NCONT + NBIN
      
C     BOUNDS FOR Q1, Q2 AND Q3      
      
       XL(1)= 0.0D0
       XL(2)= 0.0D0
       XL(3)= 0.0D0    
       XL(4)= 0.0D0
       XL(5)= 0.0D0
       XL(6)= 0.0D0
       
       XU(1)= 900.D0
       XU(2)= 800.D0
       XU(3)= 700.D0
       XU(4)= 600.D0
       XU(5)= 500.D0
       XU(6)= 400.D0
       
C     BOUNDS FOR SW1_SF1, ... SW3_SF3:
       XL(7)= 0.0D0
       XL(8)= 0.0D0
       XL(9)= 0.0D0
       XL(10)= 0.0D0
       XL(11)= 0.0D0
       XL(12)= 0.0D0
       XL(13)= 0.0D0
       XL(14)= 0.0D0
       XL(15)= 0.0D0
       XL(16)= 0.0D0
       XL(17)= 0.0D0
       XL(18)= 0.0D0
       XL(19)= 0.0D0
       XL(20)= 0.0D0
       XL(21)= 0.0D0
       XL(22)= 0.0D0
       XL(23)= 0.0D0
       XL(24)= 0.0D0
      
       XU(7)= 1.0D0
       XU(8)= 1.0D0
       XU(9)= 1.0D0
       XU(10)= 1.0D0
       XU(11)= 1.0D0
       XU(12)= 1.0D0
       XU(13)= 1.0D0
       XU(14)= 1.0D0
       XU(15)= 1.0D0
       XU(16)= 1.0D0
       XU(17)= 1.0D0
       XU(18)= 1.0D0
       XU(19)= 1.0D0
       XU(20)= 1.0D0
       XU(21)= 1.0D0
       XU(22)= 1.0D0
       XU(23)= 1.0D0
       XU(24)= 1.0D0

       FEX= -3099.5D0
       FEX=(1.0D+4+FEX)*FSCALE
      GOTO 9999  
C
C   MITP144 Shell 10 Wells modified liftgas
C
  144 CONTINUE
      PNAM= 'MITP193(SHELL 10Well-modified-17000)'
      M=11
      ME=0
      NBIN=10
      NCONT=10
      NINT=0
      N = NINT + NCONT + NBIN
      
      LGMAX=2000.D0
C     BOUNDS 
       DO I=NCONT+1,N
	  XL(I)=0.D0
	  XU(I)=1.D0
       ENDDO
       DO I=1,NCONT
        XL(I)=0.D0
       ENDDO
      
C     BOUNDS 
      XU(1)=1000.d0*SQRT((LGMAX-300.D0)/LGMAX)
      XU(2)=1100.d0*SQRT((LGMAX-350.D0)/LGMAX)
      XU(3)=1200.d0*SQRT((LGMAX-400.D0)/LGMAX)
      XU(4)=1300.d0*SQRT((LGMAX-450.D0)/LGMAX)
      XU(5)=1400.d0*SQRT((LGMAX-500.D0)/LGMAX)
      XU(6)=1500.d0*SQRT((LGMAX-550.D0)/LGMAX)
      XU(7)=1600.d0*SQRT((LGMAX-600.D0)/LGMAX)
      XU(8)=1700.d0*SQRT((LGMAX-650.D0)/LGMAX)
      XU(9)=1800.d0*SQRT((LGMAX-700.D0)/LGMAX)
      XU(10)=1900.d0*SQRT((LGMAX-750.D0)/LGMAX)
    
       
       DO I=1,NCONT
	     X0(I)=XL(I)+(XU(I)-XL(I))/2.d0
       ENDDO
       DO I=NCONT+1,N
         X0(I)=0.d0
       ENDDO
C FOR I=1,10  
       XEX(I)=1.D0
       
       
C OPTIMAL CONTINUOUS VARIABLE VALUES UNKNOWN
       FEX=-11237.8427D0
      GOTO 9999       
C
C   MITP145 Shell 10 Wells modified liftgas
C
  145 CONTINUE
      PNAM= 'MITP194(SHELL 10Well-modified-18000)'
      M=11
      ME=0
      NBIN=10
      NCONT=10
      NINT=0
      N = NINT + NCONT + NBIN
      
      LGMAX=2000.D0
C     BOUNDS 
       DO I=NCONT+1,N
	  XL(I)=0.D0
	  XU(I)=1.D0
       ENDDO
       DO I=1,NCONT
        XL(I)=0.D0
       ENDDO
      
C     BOUNDS 
      XU(1)=1000.d0*SQRT((LGMAX-300.D0)/LGMAX)
      XU(2)=1100.d0*SQRT((LGMAX-350.D0)/LGMAX)
      XU(3)=1200.d0*SQRT((LGMAX-400.D0)/LGMAX)
      XU(4)=1300.d0*SQRT((LGMAX-450.D0)/LGMAX)
      XU(5)=1400.d0*SQRT((LGMAX-500.D0)/LGMAX)
      XU(6)=1500.d0*SQRT((LGMAX-550.D0)/LGMAX)
      XU(7)=1600.d0*SQRT((LGMAX-600.D0)/LGMAX)
      XU(8)=1700.d0*SQRT((LGMAX-650.D0)/LGMAX)
      XU(9)=1800.d0*SQRT((LGMAX-700.D0)/LGMAX)
      XU(10)=1900.d0*SQRT((LGMAX-750.D0)/LGMAX)
    
       
       DO I=1,NCONT
	     X0(I)=XL(I)+(XU(I)-XL(I))/2.d0
       ENDDO
       DO I=NCONT+1,N
         X0(I)=0.d0
       ENDDO
C FOR I=1,10  
       XEX(I)=1.D0
       
       
C OPTIMAL CONTINUOUS VARIABLE VALUES UNKNOWN
       FEX=-11644.6117D0
      GOTO 9999 

C
C   MITP146 Shell 10 Wells modified liftgas
C
  146 CONTINUE
      PNAM= 'MITP195(SHELL 10Well-modified-19000)'
      M=11
      ME=0
      NBIN=10
      NCONT=10
      NINT=0
      N = NINT + NCONT + NBIN
      
      LGMAX=2000.D0
C     BOUNDS 
       DO I=NCONT+1,N
	  XL(I)=0.D0
	  XU(I)=1.D0
       ENDDO
       DO I=1,NCONT
        XL(I)=0.D0
       ENDDO
      
C     BOUNDS 
      XU(1)=1000.d0*SQRT((LGMAX-300.D0)/LGMAX)
      XU(2)=1100.d0*SQRT((LGMAX-350.D0)/LGMAX)
      XU(3)=1200.d0*SQRT((LGMAX-400.D0)/LGMAX)
      XU(4)=1300.d0*SQRT((LGMAX-450.D0)/LGMAX)
      XU(5)=1400.d0*SQRT((LGMAX-500.D0)/LGMAX)
      XU(6)=1500.d0*SQRT((LGMAX-550.D0)/LGMAX)
      XU(7)=1600.d0*SQRT((LGMAX-600.D0)/LGMAX)
      XU(8)=1700.d0*SQRT((LGMAX-650.D0)/LGMAX)
      XU(9)=1800.d0*SQRT((LGMAX-700.D0)/LGMAX)
      XU(10)=1900.d0*SQRT((LGMAX-750.D0)/LGMAX)
    
       
       DO I=1,NCONT
	     X0(I)=XL(I)+(XU(I)-XL(I))/2.d0
       ENDDO
       DO I=NCONT+1,N
         X0(I)=0.d0
       ENDDO
C FOR I=1,10  
       XEX(I)=1.D0
       
C OPTIMAL CONTINUOUS VARIABLE VALUES UNKNOWN
       FEX=-12007.6956D0
      GOTO 9999      
C
C   MITP147 Shell WellRelinking663C including compressors (medium)
C
  147 CONTINUE
      PNAM= 'MITP141(SHELL WR663C-600/600)'
      M=12
      ME=0
      NBIN=12
      NCONT=6
      NINT=0
      N = NINT + NCONT + NBIN

C     BOUNDS 
       DO I=1,N
	 XL(I)=0.D0
	 XU(I)=1.D0
       ENDDO
      
       
       DO I=1,6
	     X0(I)=XL(I)+(XU(I)-XL(I))/2.d0
       ENDDO
       
       DO I=7,18
	X0(I)=1.d0
       ENDDO

       FEX= -2634.D0
c       FEX=(1.0D+4+FEX)*FSCALE

      GOTO 9999
C
C   MITP148 Shell WellRelinking663C including compressors (medium)
C
  148 CONTINUE
      PNAM= 'MITP142(SHELL WR663C-600/400)'
      M=12
      ME=0
      NBIN=12
      NCONT=6
      NINT=0
      N = NINT + NCONT + NBIN

C     BOUNDS 
       DO I=1,N
	 XL(I)=0.D0
	 XU(I)=1.D0
       ENDDO
      
       
       DO I=1,6
	     X0(I)=XL(I)+(XU(I)-XL(I))/2.d0
       ENDDO
       
       DO I=7,18
	X0(I)=1.d0
       ENDDO

       FEX= -2546.D0
c       FEX=(1.0D+4+FEX)*FSCALE

      GOTO 9999
C
C   MITP149 Shell WellRelinking663C including compressors (medium)
C
  149 CONTINUE
      PNAM= 'MITP143(SHELL WR663C-600/300)'
      M=12
      ME=0
      NBIN=12
      NCONT=6
      NINT=0
      N = NINT + NCONT + NBIN

C     BOUNDS 
       DO I=1,N
	 XL(I)=0.D0
	 XU(I)=1.D0
       ENDDO
      
       
       DO I=1,6
	     X0(I)=XL(I)+(XU(I)-XL(I))/2.d0
       ENDDO
       
       DO I=7,18
	X0(I)=1.d0
       ENDDO

       FEX= -2499.D0
c       FEX=(1.0D+4+FEX)*FSCALE

      GOTO 9999

C
C   MITP150 Shell 10 Wells modified liftgas
C
  150 CONTINUE
      PNAM= 'MITP162(SHELL 10Well-modified-1100)'
      M=11
      ME=0
      NBIN=10
      NCONT=10
      NINT=0
      N = NINT + NCONT + NBIN
      
      LGMAX=2000.D0
C     BOUNDS 
       DO I=NCONT+1,N
	  XL(I)=0.D0
	  XU(I)=1.D0
       ENDDO
       DO I=1,NCONT
        XL(I)=0.D0
       ENDDO
      
C     BOUNDS 
      XU(1)=1000.d0*SQRT((LGMAX-300.D0)/LGMAX)
      XU(2)=1100.d0*SQRT((LGMAX-350.D0)/LGMAX)
      XU(3)=1200.d0*SQRT((LGMAX-400.D0)/LGMAX)
      XU(4)=1300.d0*SQRT((LGMAX-450.D0)/LGMAX)
      XU(5)=1400.d0*SQRT((LGMAX-500.D0)/LGMAX)
      XU(6)=1500.d0*SQRT((LGMAX-550.D0)/LGMAX)
      XU(7)=1600.d0*SQRT((LGMAX-600.D0)/LGMAX)
      XU(8)=1700.d0*SQRT((LGMAX-650.D0)/LGMAX)
      XU(9)=1800.d0*SQRT((LGMAX-700.D0)/LGMAX)
      XU(10)=1900.d0*SQRT((LGMAX-750.D0)/LGMAX)
       DO I=1,NCONT
	     X0(I)=XL(I)+(XU(I)-XL(I))/2.d0
       ENDDO
       DO I=NCONT+1,N
         X0(I)=1.d0
       ENDDO
       XEX(1)=0.D0
       XEX(2)=0.D0
       XEX(3)=0.D0
       XEX(4)=0.D0
       XEX(5)=0.D0
       XEX(6)=0.D0
       XEX(7)=0.D0
       XEX(8)=1.D0
       XEX(9)=0.D0
       XEX(10)=0.D0
       
C OPTIMAL CONTINUOUS VARIABLE VALUES UNKNOWN
       FEX=-806.3808D0
      GOTO 9999      
      
      


       
C
C   MITP151 - MITP160 (RASTRINGIN SERIES)
C
  151 DIM = 2
      GOTO 1002
  152 DIM = 3
      GOTO 1002
  153 DIM = 4
      GOTO 1002
  154 DIM = 5
      GOTO 1002
  155 DIM = 6
      GOTO 1002
  156 DIM = 7
      GOTO 1002
  157 DIM = 8
      GOTO 1002
  158 DIM = 9
      GOTO 1002
  159 DIM = 10
      GOTO 1002
  160 DIM = 11
      GOTO 1002      
 1002 CONTINUE
         PNAM = 'RASTRINGIN'
         PREF = ''
         NCONT = DIM
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -5.12D0 - DBLE(DIM)/7.0d0
             XU(I) =  5.32D0 + DBLE(DIM)/10.0D0 
              X(I) = XL(I) 
         ENDDO              
         FEX = 1.0D-5
         GOTO 9999     
C
C   MITP161 - MITP170 (GRIEWANK SERIES)
C
  161 DIM = 2
      GOTO 1003
  162 DIM = 3
      GOTO 1003
  163 DIM = 4
      GOTO 1003
  164 DIM = 5
      GOTO 1003
  165 DIM = 6
      GOTO 1003
  166 DIM = 7
      GOTO 1003
  167 DIM = 8
      GOTO 1003
  168 DIM = 9
      GOTO 1003
  169 DIM = 10
      GOTO 1003
  170 DIM = 11
      GOTO 1003      
 1003 CONTINUE
         PNAM = 'GRIEWANK'
         PREF = ''
         NCONT = DIM
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -600.12D0 - DBLE(DIM)/3.0d0
             XU(I) =  600.32D0 + DBLE(DIM)/5.0D0
              X(I) = XL(I) 
         ENDDO              
         FEX = 1.0D-5
         GOTO 9999 
C
C   MITP171 - MITP180 (SCHWEFEL SERIES)
C
  171 DIM = 2
      GOTO 1004
  172 DIM = 3
      GOTO 1004
  173 DIM = 4
      GOTO 1004
  174 DIM = 5
      GOTO 1004
  175 DIM = 6
      GOTO 1004
  176 DIM = 7
      GOTO 1004
  177 DIM = 8
      GOTO 1004
  178 DIM = 9
      GOTO 1004
  179 DIM = 10
      GOTO 1004
  180 DIM = 11
      GOTO 1004      
 1004 CONTINUE
         PNAM = 'SCHWEFEL'
         PREF = ''
         NCONT = DIM
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -500.1D0 - 1.0D0/DBLE(DIM)
             XU(I) =  500.0D0 + 2.0D0/DBLE(DIM)
              X(I) = XL(I) 
         ENDDO              
         FEX = 0.0002D0
         GOTO 9999           
C
C   MITP181 - MITP190 (ACKLEY SERIES)
C
  181 DIM = 2
      GOTO 1005
  182 DIM = 3
      GOTO 1005
  183 DIM = 4
      GOTO 1005
  184 DIM = 5
      GOTO 1005
  185 DIM = 6
      GOTO 1005
  186 DIM = 7
      GOTO 1005
  187 DIM = 8
      GOTO 1005
  188 DIM = 9
      GOTO 1005
  189 DIM = 10
      GOTO 1005
  190 DIM = 11
      GOTO 1005      
 1005 CONTINUE
         PNAM = 'ACKLEY'
         PREF = ''
         NCONT = DIM
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -15.46D0 - DBLE(DIM)/20.0D0
             XU(I) =  30.24D0 + DBLE(DIM)/51.0D0
              X(I) = XL(I) 
         ENDDO              
         FEX = 1.0D-5
         GOTO 9999                                                     
C
C   MITP191 - MITP200 (ROSENBROCK SERIES)
C
  191 DIM = 2
      GOTO 2001
  192 DIM = 3
      GOTO 2001
  193 DIM = 4
      GOTO 2001
  194 DIM = 5
      GOTO 2001
  195 DIM = 6
      GOTO 2001
  196 DIM = 7
      GOTO 2001
  197 DIM = 8
      GOTO 2001
  198 DIM = 9
      GOTO 2001
  199 DIM = 10
      GOTO 2001
  200 DIM = 11
      GOTO 2001      
 2001 CONTINUE
         PNAM = 'ROSENBROCK'
         PREF = ''
         NCONT = DIM
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -5.12D0 - DBLE(DIM)**2
             XU(I) =  6.87D0 + DBLE(DIM)*3.0D0
              X(I) = XL(I) 
         ENDDO             
         FEX = 1.0D-5
         IF(IPROB.LE.195) FEX = 1.0D-9         
         GOTO 9999  
C
C   MITP201
C       
  201 CONTINUE       
      PNAM='TP16'
      NCONT = 2
      NINT  = 0
      NBIN  = 0
      M     = 2
      ME    = 0
      N = NINT + NCONT + NBIN      
      XL(1)=-2.0
      XU(1)=0.5
      XL(1)=-1.0D+2
      XL(2)=-1.0D+1
      XU(2)=1.0
      FEX=0.25D0   
      GOTO 9999
C
C   MITP202
C       
  202 CONTINUE       
      PNAM='TP33'
      NCONT = 3
      NINT  = 0
      NBIN  = 0
      M     = 2
      ME    = 0
      N = NINT + NCONT + NBIN         
      XL(1)=0.0
      XU(1)=1.0D+1
      XL(2)=0.0
      XU(2)=1.0D+1
      XL(3)=0.0
      XU(3)=5.0
      FEX=DSQRT(2.0D0)-6.0D0  
      GOTO 9999      
C
C   MITP203
C       
  203 CONTINUE       
      PNAM='TP55'
      NCONT = 6
      NINT  = 0
      NBIN  = 0
      M     = 6
      ME    = 6
      N = NINT + NCONT + NBIN        
      DO I=1,6
      XU(I)=1.0D+2
      XU(I)=1.0D+1
      XL(I)=0.D0
      XU(1)=1.D+0
      ENDDO
      FEX=19.0D0/3.0D0  
      GOTO 9999   
C
C   MITP204
C       
  204 CONTINUE       
      PNAM='TP265'
      NCONT = 4
      NINT  = 0
      NBIN  = 0
      M     = 2
      ME    = 2
      N = NINT + NCONT + NBIN       
      DO I=1,4
          XL(I)=0.D+0 
          XU(I)=1.D+0
      ENDDO
      FEX=0.97474658D+0  
      GOTO 9999       
C
C   MITP205
C       
  205 CONTINUE       
         PNAM='Murtagh, Saunders'

          NCONT = 5
          NINT  = 0
          NBIN  = 0
          M     = 3
          ME    = 3
          N = NINT + NCONT + NBIN    
           
         DO I=1,N
         XL(I)=-5.0
         XU(I)=5.0
         ENDDO
         FEX=0.2934D-01  
      GOTO 9999 
C
C   MITP206
C       
  206 CONTINUE       
      PNAM='Pinter 1'
          NCONT = 50
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN
      DO I=1,N
      XL(I)=-10.0
      XU(I)=10.0
      ENDDO
      FEX=1.0  
      GOTO 9999   
C
C   MITP207
C       
  207 CONTINUE   
      PNAM='Hesse'
          NCONT = 6
          NINT  = 0
          NBIN  = 0
          M     = 6
          ME    = 0
          N = NINT + NCONT + NBIN      
      DO I=1,N
      XL(I)=0.0
      XU(I)=10.0
      ENDDO
      XL(3)=1.0
      XU(3)=5.0
      XU(4)=6.0
      XL(5)=1.0
      XU(5)=5.0
      XU(6)=10.0
      FEX=-310.0
      GOTO 9999 
C
C   MITP208
C       
  208 CONTINUE   
      PNAM = 'Pinter constrained 1'

          NCONT = 2
          NINT  = 0
          NBIN  = 0
          M     = 3
          ME    = 0
          N = NINT + NCONT + NBIN 
                
      XL(1) = -1.0D0
      XU(1) = 3.0D0
      XL(2) = -2.0D0
      XU(2) = 3.0D0      
      FEX = 1.0D0
      GOTO 9999
C
C   MITP209
C       
  209 CONTINUE   
      PNAM = 'Pinter constrained 2'

          NCONT = 2
          NINT  = 0
          NBIN  = 0
          M     = 4
          ME    = 0
          N = NINT + NCONT + NBIN
          
      XL(1) = -1.0D0
      XU(1) = 3.0D0
      XL(2) = -2.0D0
      XU(2) = 3.0D0      
      FEX = 1.0D0
      GOTO 9999  
C
C   MITP210
C       
  210 CONTINUE   
         PNAM='TP327'

          NCONT = 2
          NINT  = 0
          NBIN  = 0
          M     = 1
          ME    = 0
          N = NINT + NCONT + NBIN
          
         DO I=1,N
            XL(I) = 0.4D0
            XU(I) = 1.0D5
         ENDDO   
         FEX = 0.28459670D-01
      GOTO 9999
C
C   MITP211
C       
  211 CONTINUE   
          PNAM='Hedar G1'
          NCONT = 13
          NINT  = 0
          NBIN  = 0
          M     = 9
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 1.0D0
          ENDDO   
          XU(10) = 100.0D0
          XU(11) = 100.0D0
          XU(12) = 100.0D0                    
          
          FEX = -15.0D0
      GOTO 9999 
C
C   MITP212
C       
  212 CONTINUE   
          PNAM='Hedar G2'
          NCONT = 20
          NINT  = 0
          NBIN  = 0
          M     = 2
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 10.0D0
          ENDDO   
          XL(1) = 1.0D-12
          
          FEX = -0.8036D0 
      GOTO 9999
C
C   MITP213
C       
  213 CONTINUE   
          PNAM='Hedar G3'
          NCONT = 17
          NINT  = 0
          NBIN  = 0
          M     = 1
          ME    = 1
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 0.0D0 
             XU(I) = 1.0D0
          ENDDO   

          FEX = -1.0D0
      GOTO 9999 
C
C   MITP214
C       
  214 CONTINUE   
          PNAM='Hedar G4'
          NCONT = 5
          NINT  = 0
          NBIN  = 0
          M     = 6
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 27.0D0 
             XU(I) = 45.0D0
          ENDDO   
          XL(1) = 78.0D0
          XL(2) = 33.0D0
          XU(1) = 102.0D0

          FEX = -30665.539D0
      GOTO 9999
C
C   MITP215
C       
  215 CONTINUE   
          PNAM='Hedar G5'
          NCONT = 4
          NINT  = 0
          NBIN  = 0
          M     = 5
          ME    = 3
          DO I=1,N
             XL(I) = -0.55D0 
             XU(I) = 0.55D0
          ENDDO   
          XL(1) = 0.0D0
          XL(2) = 0.0D0
          XU(1) = 1200.0D0
          XU(2) = 1200.0D0          
          
          FEX = 5126.4981D0          
      GOTO 9999   
C
C   MITP216
C       
  216 CONTINUE   
          PNAM='Hedar G6'
          NCONT = 2
          NINT  = 0
          NBIN  = 0
          M     = 2
          ME    = 0
          N = NINT + NCONT + NBIN          
        
          XL(1) = 13.0D0
          XL(2) = 0.0D0
          XU(1) = 100.0D0
          XU(2) = 100.0D0          
          
          FEX = -6961.81388D0
      GOTO 9999 
C
C   MITP217
C       
  217 CONTINUE   
          PNAM='Hedar G7'
          NCONT = 10
          NINT  = 0
          NBIN  = 0
          M     = 8
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = -10.0D0 
             XU(I) = 10.0D0
          ENDDO         
          
          x(1) = 2.171996
          x(2) = 2.363683
          x(3) = 8.773926
          x(4) = 5.095984
          x(5) = 0.9906548
          x(6) = 1.430574
          x(7) = 1.321644
          x(8) = 9.828726
          x(9) = 8.280092
          x(10) = 8.375927
          
          FEX = 24.3062091D0
      GOTO 9999
C
C   MITP218
C       
  218 CONTINUE   
          PNAM='Hedar G8'
          NCONT = 2
          NINT  = 0
          NBIN  = 0
          M     = 2
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 1.0D-12
             XU(I) = 10.0D0
          ENDDO    
          
!          xl(1) = 1.2279713
!          xl(2) = 4.2453733

          FEX = -0.095825D0
      GOTO 9999  
C
C   MITP219
C       
  219 CONTINUE   
          PNAM='Hedar G9'
          NCONT = 7
          NINT  = 0
          NBIN  = 0
          M     = 4
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = -10.0D0
             XU(I) = 10.0D0
          ENDDO    
          
!          xl(1) = 2.330499
!          xl(2) = 1.951372
!          xl(3) = -0.4775414
!          xl(4) = 4.365726
!          xl(5) = -0.6244870
!          xl(6) = 1.038131
!          xl(7) = 1.594227

          FEX = 680.6300573D0
      GOTO 9999     
C
C   MITP220
C       
  220 CONTINUE   
          PNAM='Hedar G10'
          NCONT = 8
          NINT  = 0
          NBIN  = 0
          M     = 6
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 10.0D0
             XU(I) = 1000.0D0
          ENDDO    
          XL(1) = 100.0d0
          XL(2) = 1000.0D0
          XL(3) = 1000.0D0
          XU(1) = 10000.0D0
          XU(2) = 10000.0D0
          XU(3) = 10000.0D0

!         xl(1) = 579.3167
!         xl(2) =  1359.943
!         xl(3) =  5110.071
!         xl(4) = 182.0174
!         xl(5) = 295.5985
!         xl(6) = 217.9799
!         xl(7) = 286.4162
!         xl(8) = 395.5979
         
          FEX = 7049.3307D0
      GOTO 9999                
C
C   MITP221
C       
  221 CONTINUE   
          PNAM='Hedar G11'
          NCONT = 2
          NINT  = 0
          NBIN  = 0
          M     = 1
          ME    = 1
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = -1.0D0
             XU(I) = 1.0D0
          ENDDO    
          FEX = 0.75D0
      GOTO 9999   
C
C   MITP222
C       
  222 CONTINUE   
          PNAM  = 'Perm Function+Rosenbrock'
          NCONT = 12
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
           DO I=1,8
             XL(I) = -NCONT
             XU(I) =  NCONT
          ENDDO 
           DO I=9,12
             XL(I) = -43.0D0
             XU(I) =  7.0D0
          ENDDO           
          FEX = 1.0D-5
      GOTO 9999
C
C   MITP223
C       
  223 CONTINUE   
          PNAM='Hedar G13'
          NCONT = 5
          NINT  = 0
          NBIN  = 0
          M     = 3
          ME    = 3
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = -3.2D0
             XU(I) =  3.2D0
          ENDDO       
          XL(1) = -2.3D0
          XL(2) = -2.3D0
          XU(1) = 2.3D0
          XU(2) = 2.3D0
                 
          FEX = 0.0539498D0
      GOTO 9999
C
C   MITP224
C       
  224 CONTINUE   
          PNAM='Hedar Welded Beam Design Problem'
          NCONT = 4
          NINT  = 0
          NBIN  = 0
          M     = 6
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 0.1D0
             XU(I) = 10.0D0
          ENDDO       
          XL(1) = 0.125D0
                 
          FEX =   2.2182D0
      GOTO 9999  
C
C   MITP225
C       
  225 CONTINUE   
          PNAM='Hedar Pressure Vessel Design Problem'
          NCONT = 4
          NINT  = 0
          NBIN  = 0
          M     = 4
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 1.0D-12
             XU(I) = 1000.0D0
          ENDDO   
                           
          FEX =    5803.3887D0
      GOTO 9999 
C
C   MITP226
C       
  226 CONTINUE   
          PNAM  = 'SMALL DUMMY *precision*'
          NCONT = 1
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 777.0D0
          ENDDO   
          FEX = 636.3636D0
      GOTO 9999
C
C   MITP227
C       
  227 CONTINUE   
          PNAM  = 'SMALL DUMMY'
          NCONT = 0
          NINT  = 1
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 777.0D0
          ENDDO   
          FEX = 666.0D0
      GOTO 9999
C
C   MITP228
C       
  228 CONTINUE   
          PNAM  = 'SMALL DUMMY'
          NCONT = 1
          NINT  = 0
          NBIN  = 0
          M     = 1
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 777.0D0
          ENDDO   
          FEX = 666.0D0
      GOTO 9999
C
C   MITP229
C       
  229 CONTINUE   
          PNAM  = 'SMALL DUMMY'
          NCONT = 0
          NINT  = 1
          NBIN  = 0
          M     = 1
          ME    = 1
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 777.0D0
          ENDDO   
          FEX = 666.0D0
      GOTO 9999
C
C   MITP230
C       
  230 CONTINUE   
          PNAM  = 'SMALL DUMMY'
          NCONT = 1
          NINT  = 1
          NBIN  = 0
          M     = 1
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 777.0D0
          ENDDO   
          FEX = 666.0D0
      GOTO 9999
C
C   MITP231
C       
  231 CONTINUE   
          PNAM  = 'Perm Function *modified*'
          NCONT = 4
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
           DO I=1,N
             XL(I) = -10.0D0
             XU(I) =  10.0D0
          ENDDO 
          FEX = 1.0D-5
      GOTO 9999
C
C   MITP232
C       
  232 CONTINUE   
          PNAM  = 'Schwefel D10'
          NCONT = 10
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = -10.0D0
             XU(I) =  10.0D0
          ENDDO   
          FEX = -89.7931D0
      GOTO 9999   
C
C   MITP233
C       
  233 CONTINUE   
          PNAM  = 'Whitley'
          NCONT = 2
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = -10.0D0
             XU(I) =  10.0D0
          ENDDO   
          FEX = -17.6601D0
      GOTO 9999   
C
C   MITP234
C       
  234 CONTINUE   
          PNAM  = 'Bohachecsky'
          NCONT = 2
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = -100.0D0
             XU(I) =  100.0D0
          ENDDO   
          FEX = 1.0D-5
      GOTO 9999
C
C   MITP235
C       
  235 CONTINUE   
          PNAM  = 'Branin'
          NCONT = 2
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
          XL(1) = -5.0D0
          XU(1) = 10.0D0
          XL(2) = 0.0D0
          XU(2) = 15.0D0
          
          X(1) = PI
          X(2) = 12.275D0
          
          FEX = 0.397887D0
      GOTO 9999
C
C   MITP236
C       
  236 CONTINUE   
          PNAM  = 'Perm Function'
          NCONT = 10
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
           DO I=1,N
             XL(I) = -NCONT
             XU(I) =  NCONT
          ENDDO 
                 
          FEX = 1.0D-5
      GOTO 9999       
C
C   MITP237
C
  237 CONTINUE
         PNAM = 'Stairs'
         PREF = ''
         NCONT = 50
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = -1000.0D0
         XU(I) = 70000.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-5
       GOTO 9999
C
C   MITP238
C
  238 CONTINUE
         PNAM = 'Equilibrium (PAPA)'
         NCONT = 7
         NINT  = 0
         NBIN  = 0
         M     = 7
         ME    = 5
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = 1.0D-4
         XU(I) = 1.0D0
          X(I) = XL(I) 
         ENDDO
         XU(7) = 3.0D0 
         
          X(  1) =   0.3228663318833293D+00
          X(  2) =   0.9224147724768967D-02
          X(  3) =   0.4602227924958010D-01
          X(  4) =   0.6181702585493642D+00
          X(  5) =   0.3716354539215302D-02
          X(  6) =   0.5766245307121118D+00
          X(  7) =   0.2977785281822376D+01
	         
         FEX = 1.0D-4
       GOTO 9999 
C
C   MITP239
C
  239 CONTINUE
         PNAM = 'High Dim Rosenbrock'
         PREF = ''
         NCONT = 15
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = -5.23D0 - DBLE(I)/DBLE(N)
         XU(I) =  5.45D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-5
       GOTO 9999
C
C   MITP240
C
  240 CONTINUE
         PNAM = 'High Dim Rosenbrock'
         PREF = ''
         NCONT = 30
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = -5.23D0 
         XU(I) =  5.45D0 + DBLE(I)/DBLE(N)
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-5
       GOTO 9999  
C
C   MITP241
C
  241 CONTINUE
         PNAM = 'High Dim Rastringin'            
         NCONT = 15
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -5.12D0 - DBLE(I)/DBLE(N)**2
             XU(I) =  5.32D0 
              X(I) = XL(I) 
         ENDDO              
         FEX = 1.0D-5
         GOTO 9999
C
C   MITP242
C
  242 CONTINUE
         PNAM = 'High Dim Rastringin'            
         NCONT = 30
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -5.14D0
             XU(I) =  5.31D0 + DBLE(I)/DBLE(N)**2
              X(I) = XL(I) 
         ENDDO              
         FEX = 1.0D-5
         GOTO 9999
C
C   MITP243
C
  243 CONTINUE
         PNAM = 'High Dim Griewank'            
         NCONT = 15
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -600.12D0
             XU(I) =  600.36D0 
              X(I) = XL(I) 
         ENDDO              
         FEX = 1.0D-5
         GOTO 9999
C
C   MITP244
C
  244 CONTINUE
         PNAM = 'High Dim Griewank'            
         NCONT = 30
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -600.15D0
             XU(I) =  600.39D0 
              X(I) = XL(I) 
         ENDDO              
         FEX = 1.0D-5
         GOTO 9999
C
C   MITP245
C       
  245 CONTINUE   
          PNAM='Hedar G2 (small-5)'
          NCONT = 5
          NINT  = 0
          NBIN  = 0
          M     = 2
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 10.0D0
          ENDDO   
          XL(1) = 1.0D-12
          
          FEX = -6.34448885D0 
      GOTO 9999
C
C   MITP246
C       
  246 CONTINUE   
          PNAM='Hedar G2 (small-10)'
          NCONT = 10
          NINT  = 0
          NBIN  = 0
          M     = 2
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 10.0D0
          ENDDO   
          XL(1) = 1.0D-12
          
          FEX = -74.73097502D0 
      GOTO 9999
C
C   MITP247
C       
  247 CONTINUE   
          PNAM='Hedar G2 (small-15)'
          NCONT = 15
          NINT  = 0
          NBIN  = 0
          M     = 2
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 10.0D0
          ENDDO   
          XL(1) = 1.0D-12
          
          FEX = -782.44028081D0 
      GOTO 9999   
C
C   MITP248
C
  248 CONTINUE
         PNAM = 'High Dim Rosenbrock'
         PREF = ''
         NCONT = 20
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = -2.5D0
         XU(I) = 2.3D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-5
       GOTO 9999
C
C   MITP249
C
  249 CONTINUE
         PNAM = 'Rosenbrock+QP+Constraints'
         PREF = ''
         NCONT = 13
         NINT  = 0
         NBIN  = 0
         M     = 3
         ME    = M
         N = NINT + NCONT + NBIN
         DO I = 1,9
             XL(I) = -3.0D0
             XU(I) = 3.0D0
         ENDDO    
         XL(10) = -13.0D0
         XU(10) = 55.0D0     
         DO I = 11,N
             XL(I) = 0.0D0
             XU(I) = DBLE(I)**2
         ENDDO    
         FEX = 14.0D0
       GOTO 9999                
C
C   MITP250 
C
  250 CONTINUE
         PNAM = 'High Dim Rastringin'            
         NCONT = 20
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -20.14D0
             XU(I) =  7.31D0 + DBLE(I)/DBLE(N)
              X(I) = XL(I) 
         ENDDO              
         FEX = 1.0D-5
         GOTO 9999
C
C   MITP251 
C
  251 CONTINUE
         PNAM = 'High Dim Griewank *modified*'            
         NCONT = 15
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -10.14D0
             XU(I) =  500.31D0 
              X(I) = XL(I) 
         ENDDO              
         FEX = 1.0D-5
         GOTO 9999
C
C   MITP252 
C
  252 CONTINUE
         PNAM = 'Griewank'            
         NCONT = 8
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -800.14D0
             XU(I) = 1500.31D0 
              X(I) = XL(I) 
         ENDDO              
         FEX = 1.0D-5
         GOTO 9999      
C
C   MITP253
C       
  253 CONTINUE   
          PNAM  = 'Perm Function'
          NCONT = 8
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
           DO I=1,N
             XL(I) = -NCONT
             XU(I) =  NCONT
          ENDDO 
          FEX = 1.0D-5
      GOTO 9999        
C
C   MITP254 
C
  254 CONTINUE
         PNAM = 'High Dim Griewank *modified*'            
         NCONT = 25
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -50.14D0
             XU(I) =  30.31D0 
              X(I) = XL(I) 
         ENDDO              
         FEX = 1.0D-5
         GOTO 9999     
C
C   MITP255
C
  255 CONTINUE
         PNAM = 'High Dim STEP'
         PREF = ''
         NCONT = 100
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) =  -10.D0 - DBLE(N)/DBLE(I)**2
         XU(I) =  100.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-4
       GOTO 9999
C
C   MITP256
C
  256 CONTINUE
          PNAM = 'Perm Function'
          PREF = ''
          NCONT = 9
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
           DO I=1,N
             XL(I) = -50.0D0
             XU(I) = 500.0D0
          ENDDO 
          FEX = 1.0D-5
       GOTO 9999
C
C   MITP257
C
  257 CONTINUE
          PNAM = 'Perm Function'
          PREF = ''
          NCONT = 12
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
           DO I=1,N
             XL(I) = -20.0D0
             XU(I) =  15.0D0
          ENDDO 
          FEX = 1.0D-5
       GOTO 9999
C
C   MITP258
C
  258 CONTINUE
          PNAM = 'Powersum *precision*'
          PREF = ''
          NCONT = 3
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
           DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 1.0D0
          ENDDO 
          FEX = 1.0D-5
       GOTO 9999
C
C   MITP259
C
  259 CONTINUE
          PNAM = 'Powersum *normal*'
          PREF = ''
          NCONT = 8
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
           DO I=1,N
             XL(I) = 0.0D0
             XU(I) = DBLE(N)
          ENDDO 
          FEX = 1.0D-5
       GOTO 9999                              
C
C   MITP260
C
  260 CONTINUE
          PNAM = 'Trid function'
          PREF = ''
          NCONT = 30
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
           DO I=1,N
             XL(I) = -DBLE(N)**2
             XU(I) =  DBLE(N)**2
          ENDDO 
          FEX = -DBLE(N)*(DBLE(N)+4.0D0)*(DBLE(N)-1.0D0)/6.0D0
       GOTO 9999                                        
C
C   MITP261
C
  261 CONTINUE
          PNAM = 'High Precision Sphere'
          PREF = ''
          NCONT = 30
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN          
           DO I=1,N
             XL(I) = 0.0D0 - DBLE(N)/DBLE(I)**2
             XU(I) = DBLE(I) * 2 
          ENDDO        
          FEX = 1.0D-5       
       GOTO 9999 
C
C   MITP262
C
  262 CONTINUE
          PNAM = 'Large LP'
          PREF = ''
          NCONT = 200
          NINT  = 0
          NBIN  = 0
          M     = 0
          ME    = 0
          N = NINT + NCONT + NBIN 
           DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 1.0D0
          ENDDO 
          
          XL(10)  = -10.0D0
          XL(20)  = -2000.0D0          
          XL(23)  = -200000.0D0           
          XL(30)  = -30.0D0          
          XL(50)  = -50.0D0          
          XL(90)  = -900.0D0          
          XL(100) = -1000.0D0          
          XL(150) = -10000.0D0          
          XL(180) = -1000.0D0                             
                    
          FEX = 1.0D-5                   
       GOTO 9999
C
C   MITP263
C
  263 CONTINUE
         PNAM = 'Part Dim Rosenbrock'
         PREF = ''
         NCONT = 40
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = -3.7D0
         XU(I) = 5.7D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-4
       GOTO 9999 
C
C   MITP264
C
  264 CONTINUE
         PNAM = 'MIX Rosenbrock Rastringin'
         PREF = ''
         NCONT = 40
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = -2.85D0 
         XU(I) = 3.8D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-4
       GOTO 9999
C
C   MITP265
C
  265 CONTINUE
         PNAM = 'High Dim Ackley'
         PREF = ''
         NCONT = 200
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = -15.9D0
         XU(I) = 30.05D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-1
       GOTO 9999
C
C   MITP266
C
  266 CONTINUE
         PNAM = 'High Dim Griewank'
         PREF = ''
         NCONT = 150
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -123.0D0
             XU(I) =  500.0D0
              X(I) = XL(I) 
         ENDDO              
         FEX = 0.01D0
       GOTO 9999
C
C   MITP267
C
  267 CONTINUE
         PNAM = 'High Dim SCHWEFEL'
         PREF = ''
         NCONT = 30
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
             XL(I) = -500.0D0
             XU(I) =  500.0D0
              X(I) = XL(I) 
         ENDDO              
         FEX = 0.0004D0
       GOTO 9999
C
C   MITP268
C
  268 CONTINUE
         PNAM = 'Benchmark Mix'
         PREF = ''
         NCONT = 2*5
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,2 ! Rastringin
             XL(I) = -51.0D0
             XU(I) =  6.0D0
         ENDDO
         DO I = 3,4 ! Griewank
             XL(I) = -600.0D0
             XU(I) =  700.0D0
         ENDDO  
         DO I = 5,6 ! Schwefel
             XL(I) = -500.0D0
             XU(I) =  500.0D0
         ENDDO
         DO I = 7,8 ! Ackley
             XL(I) = -90.0D0
             XU(I) =  80.0D0
         ENDDO       
         DO I = 9,10 ! Rosenbrock
             XL(I) = -5.0D0
             XU(I) =  40.0D0
         ENDDO                              
         FEX = 0.0001D0
       GOTO 9999
C
C   MITP269
C
  269 CONTINUE
         PNAM = 'Benchmark Mix'
         PREF = ''
         NCONT = 3*5
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,3 ! Rastringin
             XL(I) = -5.0D0
             XU(I) =  6.0D0
         ENDDO
         DO I = 4,6 ! Griewank
             XL(I) = -60.0D0
             XU(I) =  20.0D0
         ENDDO  
         DO I = 7,9 ! Schwefel
             XL(I) = -500.0D0
             XU(I) =  500.0D0
         ENDDO
         DO I = 10,12 ! Ackley
             XL(I) = -90.0D0
             XU(I) =  80.0D0
         ENDDO       
         DO I = 13,15 ! Rosenbrock
             XL(I) = -1.0D0
             XU(I) =  4.0D0
         ENDDO                              
         FEX = 0.0001D0
       GOTO 9999
C
C   MITP270
C
  270 CONTINUE
         PNAM = 'Benchmark Mix'
         PREF = ''
         NCONT = 4*5
         NINT  = 0
         NBIN  = 0
         M     = 0
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,4 ! Rastringin
             XL(I) = -5.0D0
             XU(I) =  6.0D0
         ENDDO
         DO I = 5,8 ! Griewank
             XL(I) = -30.0D0
             XU(I) =  60.0D0
         ENDDO  
         DO I = 9,12 ! Schwefel
             XL(I) = -500.0D0
             XU(I) =  500.0D0
         ENDDO
         DO I = 13,16 ! Ackley
             XL(I) = -90.0D0
             XU(I) =  80.0D0
         ENDDO       
         DO I = 17,20 ! Rosenbrock
             XL(I) = -1.12D0
             XU(I) =  1.24D0
         ENDDO                              
         FEX = 0.0001D0
       GOTO 9999        
C
C   MITP271
C
  271 CONTINUE
         PNAM  = 'Large NLP (linear)'
         NCONT = 1000
         NINT  = 0
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = XL(I)
         ENDDO
         FEX = -1.0D0
      GOTO 9999
C
C   MITP272
C
  272 CONTINUE
         PNAM  = 'Large NLP (linear)'
         NCONT = 750
         NINT  = 0
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = 1.0D0
            X(I) = XL(I)
         ENDDO
         FEX = -3.75D0
      GOTO 9999
C
C   MITP273
C
  273 CONTINUE
         PNAM  = 'Large QP'
         NCONT = 500
         NINT  = 0
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N
            XL(I) = 0.0D0
            XU(I) = DBLE(N)
            X(I) = XL(I)
         ENDDO
         FEX = 1.0D-5
      GOTO 9999  
C
C   MITP274
C
  274 CONTINUE
         PNAM  = 'Large step'
         NCONT = 400
         NINT  = 0
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N
            XL(I) = -1.0D0
            XU(I) =  5.0D0 + 1.0D0/DBLE(I)
            X(I) = XL(I)
         ENDDO
         FEX = 1.0D-5
      GOTO 9999
C
C   MITP275
C
  275 CONTINUE
         PNAM  = 'Large QP step'
         NCONT = 300
         NINT  = 0
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N
            XL(I) = -1.0D0
            XU(I) =  1.0D0 + 1.0D0/DBLE(I)
            X(I) = XL(I)
         ENDDO
         FEX = 1.0D-5
      GOTO 9999                       
C
C   MITP276
C
  276 CONTINUE
         PNAM  = 'Large LP constrained'
         NCONT = 250
         NINT  = 0
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 50
         ME    = 0
         DO I=1,N
            XL(I) = 0.0D0 + 0.9D0/DBLE(I)
            XU(I) = 1.0D0 
            X(I) = XL(I)
         ENDDO
         FEX = - DBLE(N)
      GOTO 9999 
C
C   MITP277
C
  277 CONTINUE
         PNAM  = 'Large FAKE'
         NCONT = 5000
         NINT  = 0
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N
            XL(I) = 0.0D0 
            XU(I) = 1.0D0 
            X(I) = XL(I)
         ENDDO
         FEX = - 5.0D0
      GOTO 9999   
C
C   MITP278
C
  278 CONTINUE
         PNAM  = 'Large MIX'
         NCONT = 50
         NINT  = 0
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 2
         ME    = 2
         DO I=1,N
            XL(I) = -1.0D0 
            XU(I) = 2.0D0 
            X(I) = XL(I)
         ENDDO
         FEX = 2.0D0
      GOTO 9999                
C
C   MITP279
C       
  279 CONTINUE   
          PNAM='Hedar G12'
          NCONT = 3
          NINT  = 0
          NBIN  = 0
          M     = 729
          ME    = 0
          N = NINT + NCONT + NBIN          
          DO I=1,N
             XL(I) = 0.0D0
             XU(I) = 10.0D0
          ENDDO              
          FEX = -1.0D0
      GOTO 9999
C
C   MITP280
C       
  280 CONTINUE   
         PNAM  = 'Large MILP'
         NCONT = 100
         NINT  = 400
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N
            XL(I) = 0.0D0 
            XU(I) = 1.0D0 
            X(I) = XL(I)
         ENDDO
         FEX = -DBLE(N)
      GOTO 9999
C
C   MITP281
C       
  281 CONTINUE   
         PNAM  = 'Large MILP'
         NCONT = 250
         NINT  = 250
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N
            XL(I) = -1.0D0 
            XU(I) = 1.0D0 
            X(I) = XL(I)
         ENDDO
         FEX = 0.0001d0
      GOTO 9999
C
C   MITP282
C       
  282 CONTINUE   
         PNAM  = 'Large MINLP'
         NCONT = 50
         NINT  = 100
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N
            XL(I) = 0.0D0 
            XU(I) = 1.0D0 
            X(I) = XL(I)
         ENDDO
         FEX = 1.0D-4
      GOTO 9999
C
C   MITP283
C       
  283 CONTINUE   
         PNAM  = 'Large MINLP'
         NCONT = 150
         NINT  = 150
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N
            XL(I) = 0.0D0 
            XU(I) = 10.0D0
            X(I) = XL(I)
         ENDDO
         FEX = -1500.0D0
      GOTO 9999 
C
C   MITP284
C       
  284 CONTINUE   
         PNAM  = 'Large MINLP constrained'
         NCONT = 100
         NINT  = 100
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 10
         ME    = 0
         DO I=1,N
            XL(I) = -DBLE(I)/7.0D0 
            XU(I) = DBLE(I)/9.0D0
            X(I) = XL(I)
         ENDDO
         FEX = 1.0D-4
      GOTO 9999 
C
C   MITP285
C       
  285 CONTINUE   
         PNAM  = 'Large MINLP constrained'
         NCONT = 100
         NINT  = 50
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 1
         ME    = 1
         DO I=1,N
            XL(I) = 0.0D0 
            XU(I) = DBLE(I)
            X(I) = XL(I)
         ENDDO
         FEX = 1.0D-4
      GOTO 9999 
C
C   MITP286
C       
  286 CONTINUE   
         PNAM  = 'Rosen + Large MINLP'
         NCONT = 30
         NINT  = 100
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N
            XL(I) = 0.0D0 
            XU(I) = 1.0D0
            X(I) = XL(I)
         ENDDO
         FEX = 1.0D-4
      GOTO 9999                                          
C
C   MITP287 (KNAPSACK)
C
  287 CONTINUE
         PNAM = 'KNAPSACK'
         PREF = ''
         NCONT = 0
         NINT  = 0
         NBIN  = 100
         M     = 1
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = 0.0D0
         XU(I) = 1.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = -5.1072D0
         GOTO 9999
C
C   MITP288 (KNAPSACK)
C
  288 CONTINUE
         PNAM = 'KNAPSACK'
         PREF = ''
         NCONT = 0
         NINT  = 0
         NBIN  = 150
         M     = 1
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = 0.0D0
         XU(I) = 1.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = -2.9972D0
         GOTO 9999
C
C   MITP289 (KNAPSACK)
C
  289 CONTINUE
         PNAM = 'KNAPSACK'
         PREF = ''
         NCONT = 0
         NINT  = 0
         NBIN  = 250
         M     = 1
         ME    = 0
         N = NINT + NCONT + NBIN
         DO I = 1,N
         XL(I) = 0.0D0
         XU(I) = 1.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = -1.5291D0
         GOTO 9999                      
C
C   MITP290
C       
  290 CONTINUE   
         PNAM  = 'binary bomb'
         NCONT = 0
         NINT  = 10000
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N
            XL(I) = 0.0D0 
            XU(I) = 1.0D0
            X(I) = XL(I)
         ENDDO
         FEX = -DBLE(N)
      GOTO 9999
C
C   MITP291
C       
  291 CONTINUE   
         PNAM  = 'large IP'
         NCONT = 0
         NINT  = 2000
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N/2
            XL(I) = 0.0D0 
            XU(I) = 10.0D0
            X(I) = XL(I)
         ENDDO
         DO I=N/2+1,N
            XL(I) = 0.0D0 
            XU(I) = 1.0D0
            X(I) = XL(I)
         ENDDO         
         FEX = -11000.0D0
      GOTO 9999      
C
C   MITP292
C       
  292 CONTINUE   
         PNAM  = 'binary'
         NCONT = 0
         NINT  = 5000
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 0
         ME    = 0
         DO I=1,N
            XL(I) = 0.0D0 
            XU(I) = 1.0D0
            X(I) = XL(I)
         ENDDO
         FEX = -2500.0D0
      GOTO 9999
C
C   MITP293
C       
  293 CONTINUE   
         PNAM  = 'binary constrained'
         NCONT = 0
         NINT  = 1000
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 100
         ME    = 3
         DO I=1,N
            XL(I) = 0.0D0 
            XU(I) = 1.0D0
            X(I) = XL(I)
         ENDDO
         FEX = -1000.0D0
      GOTO 9999
C
C   MITP294
C       
  294 CONTINUE   
         PNAM  = 'IP constrained'
         NCONT = 0
         NINT  = 500
         NBIN  = 0
         N     = NINT + NCONT + NBIN
         M     = 100
         ME    = 1
         DO I=1,N
            XL(I) = 0.0D0 
            XU(I) = 5.0D0
            X(I) = XL(I)
         ENDDO
         FEX = -2500.0D0
      GOTO 9999            
C
C   MITP295 (128 binary)
C
  295 CONTINUE
         PNAM = 'binary'
         PREF = ''
         NCONT = 0
         NINT  = 0
         NBIN  = 128
         N = NINT + NCONT + NBIN         
         M     = N/4
         ME    = N/4
         DO I = 1,N
         XL(I) = 0.0D0
         XU(I) = 1.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = N/4
       GOTO 9999
C
C   MITP296 (256 binary)
C
  296 CONTINUE
         PNAM = 'binary'
         PREF = ''
         NCONT = 0
         NINT  = 0
         NBIN  = 256
         N = NINT + NCONT + NBIN         
         M     = N/4
         ME    = N/4
         DO I = 1,N
         XL(I) = 0.0D0
         XU(I) = 1.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = N/4
       GOTO 9999                   
C
C   MITP297 (512 binary)
C
  297 CONTINUE
         PNAM = 'binary'
         PREF = ''
         NCONT = 0
         NINT  = 0
         NBIN  = 512
         N = NINT + NCONT + NBIN         
         M     = N/4
         ME    = N/4
         DO I = 1,N
         XL(I) = 0.0D0
         XU(I) = 1.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = N/4
       GOTO 9999  
C
C   MITP298 
C
  298 CONTINUE
         PNAM = 'Fun-NLP'
         PREF = ''
         NCONT = 10
         NINT  = 0
         NBIN  = 0
         N = NINT + NCONT + NBIN         
         M     = N-1
         ME    = N/2
         DO I = 1,N
         XL(I) = -DBLE(I) * 6.0D0
         XU(I) = DBLE(I) * 10.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-5
       GOTO 9999
C
C   MITP299 
C
  299 CONTINUE
         PNAM = 'Fun-IP'
         PREF = ''
         NCONT = 0
         NINT  = 50
         NBIN  = 0
         N = NINT + NCONT + NBIN         
         M     = N-1
         ME    = N/2
         DO I = 1,N
         XL(I) = -DBLE(I) * 3.0D0
         XU(I) = DBLE(I) * 5.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-5
       GOTO 9999   
C
C   MITP300 
C
  300 CONTINUE
         PNAM = 'Fun-MINLP'
         PREF = ''
         NCONT = 10
         NINT  = 10
         NBIN  = 0
         N = NINT + NCONT + NBIN         
         M     = N-1
         ME    = N/2
         DO I = 1,N
         XL(I) = -DBLE(I) * 2.0D0
         XU(I) = DBLE(I) * 3.0D0
          X(I) = XL(I) 
         ENDDO
         FEX = 1.0D-5
       GOTO 9999               
   
  
C            
C  End of function evaluations
C
 9999 CONTINUE         
C
C  SORTING: continuous - binary - integer
C
      N = NINT + NCONT + NBIN
      IF (NINT.GT.0) THEN
         DO I=1,NINT
            X(N-I+1)  = X(NCONT+NINT-I+1)
            XL(N-I+1) = XL(NCONT+NINT-I+1)
            XU(N-I+1) = XU(NCONT+NINT-I+1)
         ENDDO
      ENDIF   
      IF (NBIN.GT.0) THEN
         DO I=1,NBIN
            X(NCONT+I)  = 1.0D0
            XL(NCONT+I) = 0.0D0
            XU(NCONT+I) = 1.0D0
         ENDDO
      ENDIF   
C
      RETURN
      END
C
      
C
C   End of SETUP
C
