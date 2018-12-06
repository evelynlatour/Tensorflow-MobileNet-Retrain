
/* Mutually Exclusive Classes for Tops: SLEEVE */
export const topSleeveClassKey = {
  shortSleeve: 0,
  longSleeve: 1,
  noSleeve: 2,
  strapless: 3
}

/* Mutually Exclusive Classes for Tops: CATEGORY */
export const topCategoryClassKey = {
  shirt: 0,
  tank: 1,
  blouse: 2,
  collaredButton: 3,
  sweater: 4,
  sweatshirt: 5,
  cardigan: 6,
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
