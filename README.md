# EffFD: Effective Flare Frequency Distribution Software

## Concept
“EffFD” will be a software package that works with light curves (observations of stellar light at regular time intervals [e.g. 20 sec]) from the Transiting Exoplanet Survey Satellite (TESS; a mission that observed many stars across the entire sky over the last few years). In most cases, these light curves look like sin waves with scattered temporary rises, which indicate that a stellar flare happened on the star (similar to the solar flares you might hear about in the news). Our plan is to have “EffFD” automatically find those flares using simple de-trending methods, return a table of the flares (with properties like duration, energy, etc.), and create flare-frequency distributions (FFDs), which are graphs that determine how often a flare of a certain energy or larger would appear. The FFDs will then be compared between different types of stars (mostly based on size) to see if there is an underlying trend.

## Reason
Flare frequency distributions are useful for various groups in astronomy. On the stellar physics side, it helps us understand the rate at which stars expel energy. This can be expanded to studying the statistical differences between different types of stars. On the exoplanet studies side, flaring rates help determine if a stellar system could be habitable or help theorize the current state of the planets there.

On software, many astronomers write their own codes from scratch for every project, which leads to work being unreproducible and unstandardized. Occasionally, flare-finding software is released, but they increasingly tend to rely on complicated Bayesian statistics and post-data modeling (e.g. Gaussian fitting), which may not be representative of the actual data. Many groups refuse to use these programs and end up relying on by-eye estimations. While the ‘preferred ways’ are still heavily debated, we believe a program that uses understandable de-trending/flare-finding methods (or at least some of the simpler ones in the field) would work well for many purposes.

## Current To-do
- Need to figure out what the index means in search target pixel file. Could use download all but takes forever, should be a user input but idk what it is.
    - Pretty sure it's the different times the star was observed. We'll have to loop through them and add the result tables together, though we could probably keep the figures separate. We could also work on the LC separately and combine the numbers at the point of making the FFD (combine flare energy tables and add together total time)
- Need to put astroquery search for list of stars by temperature range. Use the updated TESS catalogue "IV/39/tic82". Is there a qualifier for 'flaring' stars? It would cut the run time down a lot. If so, we should leave it as an option though for people who want to look at non-flaring stars to see if anything comes up. Not all stars have Teff, so we might want to find some metric of what percentage of the catalogue this covers.
    - We should use the identifiers in the 'TIC' column. 'GAIA' name might also be useful, if we need to cross-reference a star later on, but I'm not sure how we would store it neatly.
- Do we need to split the current functions up more? Especially `generate_ffd()`?
- What could be some functional and unit tests?

## Potential To-do
- We have a general way of selecting the middle regime of the FFD, but it would be better to test this against real data to see how it holds up and possibly figure out a more rigorous algorithm. This one is decently simple to explain, though, so if tests against real data come out fine, it could be something to stick with.
- ISSUE: search_lightcurve does not seem to work for all stars. AF Psc has a light curve in the MAST archive, but it doesn't return results, even if you remove the exptime and author/mission qualifiers. Need to look through lk docs and see if they only search certain TESS sectors, before a certain date, etc. I'm not sure if there's anything we can actually do about this.
- Some people might want to use just the easy FFD creation aspect with more complex flare finding routines. We could add a functionality to feed in just the property tables (e.g. star_01_flares.ecsv) and have figures created.
- What about having a function that auto-gens a config file if one doesn't exist? It could have all the options, with most commented out. If we can get the same 'args' functionality from ConfigParser, we might not need the docopt part?
    - Actually, I think I like it as-is more. This way people can't accidentally get rid of the defaults where they are necessary. I think that would be possible if there was a default config file that could be changed.
