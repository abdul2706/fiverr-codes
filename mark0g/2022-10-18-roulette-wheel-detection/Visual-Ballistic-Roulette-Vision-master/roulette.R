
# experiment 1
BALL = c(0.68, 1.20, 1.72, 2.28, 2.84, 3.40, 4.04, 4.68, 5.36, 6.04, 6.84, 7.64, 8.56, 9.64, 11.08, 12.80, 14.68, 16.76, 18.96)
plot(diff(BALL), type='o')

real_BALL_1 = c(01.049, 01.700, 02.400, 03.133, 03.916, 04.733, 05.649, 06.349, 08.266, 09.816, 11.683, 13.733)
real_BALL_2 = c(01.200, 02.133, 03.200, 04.633, 06.349, 08.249, 10.366)
real_BALL_3 = c(00.966, 01.816, 02.833, 04.066, 05.666, 07.449, 09.416, 11.633)
real_BALL_4 = c(01.349, 01.883, 02.466, 03.049, 03.666, 04.333, 05.016, 05.733, 06.533, 07.366, 08.266, 09.400, 10.833, 12.533, 14.400, 16.466)
real_BALL_5 = c(01.200, 01.716, 02.233, 02.816, 03.416, 04.016, 04.649, 05.383, 06.066, 06.833, 07.649, 08.566, 09.683, 11.066, 12.700, 14.549, 16.583, 18.683) # ca tape le diamond. quasiment un tour complet.
real_BALL_6 = c(00.666, 01.166, 01.700, 02.233, 02.800, 03.383, 03.999, 04.666, 05.333, 06.033, 06.800, 07.633, 08.566, 09.649, 11.066, 12.766, 14.649, 16.733, 19.049)
real_BALL_7 = c(01.783, 02.383, 02.983, 03.583, 04.300, 04.983, 05.749, 06.566, 07.433, 08.416, 09.733, 11.283, 13.016, 14.983, 17.149, 19.200)
real_BALL_8 = c(01.616, 02.216, 02.833, 03.533, 04.216, 04.966, 05.766, 06.666, 07.649, 08.933, 10.500, 12.349, 14.266, 16.466)
real_BALL_9 = c(00.800, 01.200, 01.633, 02.100, 02.533, 03.016, 03.466, 03.999, 04.566, 05.133, 05.700, 06.300, 06.933, 07.600, 08.316, 09.033, 09.833, 10.733, 11.716, 12.949, 14.500, 16.300, 18.200, 20.416)
real_BALL_10 = c(01.333, 01.800, 02.249, 02.749, 03.249, 03.783, 04.349, 04.949, 05.549, 06.183, 06.833, 07.549, 08.249, 09.083, 09.983, 10.916, 12.133, 13.600, 15.333, 17.249, 19.416, 21.683)
real_BALL_11 = c(00.616, 01.166, 01.783, 02.416, 03.100, 03.800, 04.533, 05.349, 06.216, 07.233, 08.549, 10.149, 11.916, 13.933, 15.966)
real_BALL_12 = c(00.600, 01.316, 02.033, 02.849, 03.733, 04.716, 05.900, 07.483, 09.216, 11.200, 13.400)
real_BALL_13 = c(02.616, 03.249, 03.916, 04.600, 05.333, 06.116, 06.949, 07.916, 09.049, 10.483, 12.166, 14.100, 16.200)
real_BALL_14 = c(00.800, 01.349, 01.949, 02.566, 03.233, 03.883, 04.666, 05.433, 06.316, 07.266, 08.466, 09.983, 11.716, 13.616, 15.783, 18.083)
real_BALL_15 = c(01.183, 01.700, 02.249, 02.833, 03.416, 04.033, 04.666, 05.383, 06.083, 06.883, 07.716, 08.666, 09.816, 11.233, 12.900, 14.783, 16.900)
real_BALL_16 = c(01.316, 01.900, 02.483, 03.100, 03.733, 04.433, 05.116, 05.866, 06.649, 07.500, 08.449, 09.649, 11.100, 12.833, 14.766, 16.866)
real_BALL_17 = c(01.100, 01.766, 02.433, 03.116, 03.866, 04.666, 05.549, 06.583, 07.849, 09.483, 11.366, 13.400)
real_BALL_18 = c(00.683, 01.183, 01.683, 02.233, 02.800, 03.383, 03.983, 04.616, 05.300, 06.033, 06.800, 07.600, 08.449, 09.466, 10.766, 12.366, 14.166, 16.200, 18.383)
real_BALL_19 = c(00.833, 01.266, 01.700, 02.133, 02.633, 03.100, 03.600, 04.149, 04.683, 05.249, 05.849, 06.466, 07.116, 07.800, 08.483, 09.266, 10.049, 10.966, 11.983, 13.266, 14.866, 16.666, 18.666, 20.816)
real_BALL_20 = c(00.683, 01.066, 01.516, 01.983, 02.500, 02.999, 03.500, 04.066, 04.633, 05.233, 05.849, 06.483, 07.183, 07.883, 08.649, 09.449, 10.366, 11.366, 12.649, 14.233, 16.016, 17.983, 20.100)
real_BALL_21 = c(00.783, 01.333, 01.949, 02.583, 03.233, 03.900, 04.649, 05.449, 06.266, 07.200, 08.266, 09.716, 11.400, 13.233, 15.316, 18.16) # 18.16 is for compatibility. The ball is off the track but passes the landmark.
real_BALL_22 = c(01.633, 02.183, 02.800, 03.416, 04.066, 04.783, 05.549, 06.333, 07.233, 08.233, 09.466, 11.033, 12.833, 14.766, 16.966)
real_BALL_23 = c(01.666, 02.183, 02.716, 03.283, 03.933, 04.516, 05.183, 05.849, 06.583, 07.400, 08.233, 09.183, 10.266, 11.766, 13.433, 15.333, 17.449)
real_BALL_24 = c(00.649, 01.133, 01.616, 02.083, 02.616, 03.133, 03.666, 04.233, 04.849, 05.466, 06.133, 06.833, 07.583, 08.333, 09.216, 10.200, 11.383, 12.866, 14.600, 16.483, 18.633)
real_BALL_25 = c(00.816, 01.283, 01.833, 02.366, 02.900, 03.466, 04.066, 04.700, 05.383, 06.066, 06.833, 07.633, 08.483, 09.533, 10.783, 12.416, 14.183, 16.149, 18.266)
real_BALL_26 = c(00.833, 01.383, 01.916, 02.516, 03.116, 03.783, 04.433, 05.166, 05.866, 06.683, 07.566, 08.566, 09.800, 11.400, 13.133, 15.183, 17.316)
real_BALL_27 = c(01.066, 01.666, 02.283, 02.916, 03.516, 04.233, 05.016, 05.816, 06.783, 07.849, 09.333, 11.033, 12.933, 15.283)
# id, deterministic_number, number, time
# 1 29 15 15.616
# 2 2 24 11.949
# 3 22 23 13.033
# 4 13 20 17.783
# 5 15 10 18.749
# 6 32 33 18.349
# 7 31 19 18.233
# 8 16 5 16.700
# 9 15 33 21.749
# 10 14 12 21.116
# 11 27 26 16.333
# 12 23 33 14.733
# 13 9 36 17.716
# 14 24 11 18.233
# 15 34 35 18.733
# 16 20 8 18.466
# 17 14 30 15.300
# 18 32 24 19.800
# 19 8 33 22.600
# 20 11 14 20.333
# 21 6 20 15.016
# 22 15 31 18.333
# 23 0 25 19.233
# 24 14 31 20.249
# 25 29 5 18.533
# 26 22 26 18.283
# 27 0 34 16.849

# ==> 1.mp4.txt <==
BALL = c(1.08, 1.72, 2.4, 3.12, 3.92, 4.76, 5.68, 6.76, 8.16, 9.84, 11.72, 13.76)
WHEEL = c(3.28, 8.44, 13.64, 21.68)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 10.mp4.txt <==
BALL = c(1.36, 1.84, 2.28, 2.76, 3.8, 4.36, 4.96, 5.56, 6.16, 6.84, 7.56, 8.28, 9.12, 9.96, 10.88, 12.16, 13.64, 15.36, 17.28, 19.44, 21.64)
WHEEL = c(4.2, 8.76, 13.36, 18., 22.68)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 11.mp4.txt <==
BALL = c(0.64, 1.2, 1.8, 2.44, 3.08, 4.56, 5.36, 6.24, 7.24, 8.56, 10.16, 11.96, 13.96, 16.08)
WHEEL = c(0.12, 4.48, 8.92, 13.36, 17.92, 22.56)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 12.mp4.txt <==
BALL = c(0.6, 1.36, 2.04, 2.88, 3.76, 4.72, 5.88, 7.52, 9.24, 11.2, 13.44)
WHEEL = c(0.4, 5.2, 9.96, 14.84, 19.76, 24.8 )
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 13.mp4.txt <==
BALL = c(2.64, 3.24, 3.96, 4.6, 5.36, 6.16, 6.96, 7.88, 9.04, 10.48, 12.2, 14.08, 16.24)
WHEEL = c(0.64, 5.16, 9.72, 14.28, 18.92, 23.6 )
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 14.mp4.txt <==
BALL = c(0.76, 1.36, 1.96, 2.56, 3.92, 4.68, 5.44, 6.32, 7.28, 8.48, 10., 11.76, 13.64, 15.8, 18.04)
WHEEL = c(2.16, 6.88, 11.56, 16.4, 21.2 )
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 15.mp4.txt <==
BALL = c(1.2, 1.72, 2.28, 2.84, 3.44, 4.04, 4.68, 5.4, 6.12, 6.92,  7.72, 8.68, 9.84, 11.28, 12.96, 14.84, 16.92)
WHEEL = c(2.16, 6.24, 10.4, 14.6, 18.8, 23.12)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 16.mp4.txt <==
BALL = c(1.36, 1.92, 2.48, 3.12, 3.76, 4.4, 5.12, 5.84, 6.68, 7.52,  8.48, 9.68, 11.2, 12.88, 14.84, 16.92)
WHEEL = c(1., 5.68, 10.36, 15.16, 20.04)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 17.mp4.txt <==
BALL = c(1.12, 1.8, 2.44, 3.12, 3.88, 4.68, 5.56, 6.6, 7.92, 9.56,  11.36, 13.04, 13.44)
WHEEL = c(3.36, 8.16, 13., 17.96, 22.92)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 18.mp4.txt <==
BALL = c(0.72, 1.2, 1.72, 2.24, 2.8, 3.4, 4., 4.64, 5.32, 6.08,  6.8, 7.64, 8.48, 9.48, 10.76, 12.4, 14.2, 16.24, 18.44)
WHEEL = c(2.72, 7.64, 12.56, 17.64, 22.68)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 19.mp4.txt <==
BALL = c(1.28, 1.72, 2.16, 3.12, 3.64, 4.12, 4.68, 5.28, 5.96, 6.48,  7.12, 7.84, 8.52, 9.28, 10.08, 10.96, 12., 13.28, 14.92, 16.72,  18.72, 20.88)
WHEEL = c(3.48, 8.36, 13.32, 18.32, 23.44)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 2.mp4.txt <==
BALL = c(0.56, 1.2, 2.12, 3.24, 3.88, 4.64, 5.76, 6.36, 7., 8.24,  8.84, 10.36, 10.84, 11.64, 13.76, 14.44)
WHEEL = c(0.08, 5.48, 11.08)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 20.mp4.txt <==
BALL = c(0.64, 1.08, 1.56, 2.48, 3.52, 4.08, 4.68, 5.28, 5.88, 6.52,  7.2, 7.92, 8.68, 9.48, 10.36, 11.36, 12.68, 14.28, 16.04, 18.04,  20.24)
WHEEL = c(3.16, 7.88, 12.72, 17.52, 22.52)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 21.mp4.txt <==
BALL = c(0.8, 1.36, 1.96, 2.56, 3.28, 3.92, 4.68, 5.48, 6.28, 7.24,  8.28, 9.68, 11.44, 13.28, 15.36, 18.08)
WHEEL = c(3.8, 8.28, 12.8, 17.4, 22.04)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 22.mp4.txt <==
BALL = c(2.84, 3.44, 4.08, 4.8, 5.56, 6.36, 7.28, 8.24, 9.48, 11.08,  12.84, 14.8, 17.04)
WHEEL = c(3.68, 7.96, 12.28, 16.64, 21.08)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 23.mp4.txt <==
BALL = c(1.68, 2.2, 2.76, 3.32, 3.92, 4.52, 5.2, 5.88, 6.6, 7.44,  8.28, 9.2, 10.32, 11.76, 13.48, 15.36, 17.48)
WHEEL = c(4.32, 8.88, 13.4, 18.08, 22.8 )
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 24.mp4.txt <==
BALL = c(0.68, 1.12, 1.64, 2.64, 3.68, 4.28, 4.88, 5.48, 6.12, 6.88,  7.56, 8.36, 9.24, 10.24, 11.36, 12.88, 14.6, 16.52, 18.68)
WHEEL = c(0.84, 4.96, 9.08, 13.32, 17.52, 21.88)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 25.mp4.txt <==
BALL = c(0.84, 1.32, 2.4, 2.92, 3.48, 4.08, 4.72, 5.4, 6.08, 6.84,  7.64, 8.52, 9.52, 10.8, 12.44, 14.16, 16.16, 18.32)
WHEEL = c(3.48, 8.2, 13., 17.84, 22.8 )
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 26.mp4.txt <==
BALL = c(0.84, 1.4, 1.96, 2.52, 3.12, 3.76, 4.48, 5.2, 5.88, 6.68,  7.56, 8.56, 9.84, 11.44, 13.24, 15.16, 17.36)
WHEEL = c(3.64, 8.68, 13.76, 19.08, 24.4 )
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 27.mp4.txt <==
BALL = c(1.08, 1.68, 2.28, 2.92, 4.24, 5.04, 5.6, 5.84, 6.76, 7.48,  7.88, 9.36, 9.84, 10.48, 11.08, 13.04, 15.32)
WHEEL = c(0.6, 5.24, 10.04, 15.16, 20.12)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 3.mp4.txt <==
BALL = c(0.96, 1.84, 2.84, 4.08, 5.68, 7.48, 9.44, 11.64)
WHEEL = c(4.4, 9.44, 14.64)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 4.mp4.txt <==
BALL = c(0.88, 1.36, 1.88, 2.48, 3.08, 3.68, 4.36, 5.04, 5.76, 6.56,  7.36, 8.28, 9.4, 10.84, 12.16, 12.52, 14.44, 15.04, 16.48, 16.88,  17.16, 18., 19.12, 19.84)
WHEEL = c(2.72, 7.48, 12.12, 17.32)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 5.mp4.txt <==
BALL = c(0.16, 1.2, 2.24, 2.8, 4.04, 4.68, 5.32, 6.08, 6.8, 7.64,  8.04, 8.6, 9.68, 11.08, 12.68, 13.8, 14.52, 16.16, 16.6, 18.88, 21.)
WHEEL = c(4.52, 9.16, 13.96, 18.76)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 6.mp4.txt <==
BALL = c(0.48, 0.68, 1.2, 1.72, 2.28, 2.84, 3.4, 4.04, 4.68, 5.36,  6.04, 6.84, 7.64, 7.96, 8.56, 9.68, 10.24, 11.08, 12., 12.8,  14.68, 16.76, 19.12, 19.76, 21.48)
WHEEL = c(2.96, 7.56, 12.2, 16.88, 21.68)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 7.mp4.txt <==
BALL = c(1.8, 2.4, 3., 3.6, 4.28, 5., 5.76, 6.56, 7.44, 8.44, 9.,  9.72, 11.28, 12., 13.04, 13.84, 14.44, 15., 16.88, 17.2, 17.6,  18.76, 19.24, 20.76)
WHEEL = c(4.4, 9.2, 14., 18.96)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 8.mp4.txt <==
BALL = c(1.64, 2.24, 2.84, 4.2, 5.76, 6.64, 7.68, 8.96, 10.52, 12.36, 14.28, 16.52)
WHEEL = c(1.72, 6.08, 10.52, 15.08, 19.64, 24.24)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')


# ==> 9.mp4.txt <==
BALL = c(1.24, 2.56, 3.48, 4.6, 5.16, 5.68, 6.28, 7.6, 9.04, 9.84,  10.76, 11.68, 13., 14.52, 16.28, 18.2, 20.4 )
WHEEL = c(2.8, 7., 11.36, 15.72, 20.12, 24.64)
plot(diff(BALL), type='o')
plot(diff(WHEEL), type='o')



plot(c( 1.8,  2.4,  3.,   3.6,  4.28, 5.,   5.76, 6.6,  7.48, 8.44, 9.72, 11.28, 13.04, 15.,  17.2, 19.24))