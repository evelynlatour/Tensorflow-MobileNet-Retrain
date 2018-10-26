
/* Mutually Exclusive Classes for Tops: SLEEVE */
const short_sleeve = 'short_sleeve'
const long_sleeve = 'long_sleeve'
const no_sleeve = 'no_sleeve'
const strapless = 'strapless'

export const topSleeveClassKey = {
  shortSleeve: 0,
  longSleeve: 1,
  noSleeve: 2,
  strapless: 3
}

/* Mutually Exclusive Classes for Tops: TYPE */
export const topTypeClassKey = {
  shirt: 0,
  sweater: 1,
  sweatshirt: 2,
  hoodie: 3,
  blouse: 4,
  collared: 5,
  cardigan: 6,
  blazer: 7,
  duster: 8
}

/* Mutually Exclusive Classes for Tops: FIT */
export const topFitKey = {
  tight: 0,
  fitted: 1,
  loose: 2,
  draped: 3
}

/* Mutually Exclusive Classes for Tops: LENGTH */
export const topLengthKey = {
  cropped: 0,
  hip: 1,
  tunic: 2,
  bodysuit: 3
}



/* Output if single multiclass NN for each - same image will be trained on multiple models 
SLEEVE [%,%,%,%] -> where each probability is related, sum to 100%
TYPE [%,%,%,%,%,%,%,%,%]
etc.
*/

/*
Multi-output --> this might actually just be called a multi-headed model
Other relevant terms - multi-task, multi-target, hard parameter sharing
But if we could take advantage of one image w/ multiple labels such that for each image output
might look like:
[[%,%,%],
[%,%,%,%,%,%,%,%],
[%,%,%,%],
[%,%,%,%]]
*/
