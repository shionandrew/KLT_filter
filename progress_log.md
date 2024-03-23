Single localization, using event id 307063854 (B0136+57 w/20º sep),306336576,304499017 (B2310+42 w/4ª sep), 268914678 (B0531+21) CHIME-KKO localization was off by over an arcsecond. 
304116324 (B1642-03) had high cal sep (ionosphere)

307063854 had a low ish snr (~20 coherent), but rfi flagging did not affect on pointing. For all sources, on pointing did really affect residual delay measured. 

It does seem like we never really get much signal at the top half of the band (or really above 500Mhz) and the localizations for these seem to improve when that half is removed. The errors seem incorrect...

running 304116324 (B1642-03) to investigate high cal separation...
still looked like rfi flagging had no effect...

-296928734 rfi heavy day; no effect....


-304499017+307063854: doing off pointing to see if this makes a difference in the localization. 
-check to see if for the above events any of the kko frequency channels were flagged...

-------------
injections
-------------

Test gains with real data!!

interpolating gains works with simulated data (of course). Polarization seems to work too with simulated data. 

need to redo the gains; run rfi filter and then calculate gains!! 

check gains: once polarimetry ends run on same event but only half of frequency channels

add polarimetry option (running, once done check that it worked)
V=np.sqrt(p**2*I-U**2)
Ex_Ex*=(I+Q)/2
Ey_Ey*=(I-Q)/2
Ey_Ex*=(U+1j*V)/2
Ex_Ey*=(U-1j*V)/2

add ionosphere term: technically implemented, need to run code to test. 


Changed amplitude of injections so that it is constant across frequency channels (otherwise we will always see it in rfi contaminated channels...)

quantify how much off pointing matters in signal matrix

refactor code and make unit tests? 
