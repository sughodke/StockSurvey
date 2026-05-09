# Use self-supervised learning to forecast the CWT

We are use patches hold out on the CWT and asking the SSL to guess what they
are. This should now apply to future patches as well. Our val scores of SSL were
okay, not amazing. But it does mean that we have learned some structure of the CWT.
