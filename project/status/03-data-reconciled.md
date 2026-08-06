## There three possibiilities:
1. xmp is available with pairing jpg; this is the best situation: photo was shot raw.
2. xmp is available but no raw or other form of photo; xmp is deleted; this situation is now clear.
3. jpg is available (by export), but no corresponding xmp. Most likely, the photo was not shot raw.

## There might still be some discripencies, but hard to fix. 
    * like the photos shot on 2021-03-28 which have matching xmp and jpg but still reported issues. 

## All jpg will be processed. 
    * If xmp is available, label is stored in xmp
    * if xmp is not available, lable is saved in csv (already implemented)

## Data audit is now clean 
    * Data audit run is now clean with zero occurrence of the possibilities 2 above
    *- for data in the year of 2016-2025