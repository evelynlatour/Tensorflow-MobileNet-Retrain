import { predictFromTruncated } from './lib/mobileNet'
import { getXs, getYs } from './lib/createXYs'
import { trainShuffledBatches, compileModel, fitModel } from './lib/trainNewModel'
import * as tf from '@tensorflow/tfjs'

import {
  predict,
  listModelsInLocalStorage,
  loadCustomModel,
  removeModelFromLocalStorage,
} from './lib/manageLocalModels'

import {
  // longSleeveTrainingData,
  // longSleeveDataLabels,
  // noSleeveTrainingData,
  // noSleeveDataLabels,
  // shortSleeveTrainingData,
  // shortSleeveDataLabels,
  // straplessTrainingData,
  // straplessDataLabels,
  blouseTrainingData,
  blouseDataLabels,
  cardiganTrainingData,
  cardiganDataLabels,
  collaredButtonTrainingData,
  collaredButtonDataLabels,
  shirtTrainingData,
  shirtDataLabels,
  sweaterTrainingData,
  sweaterDataLabels,
  sweatshirtTrainingData,
  sweatshirtDataLabels,
  tankTrainingData,
  tankDataLabels,
} from '../training-data'

import { testData } from '../test-data'

import { topSleeveClassKey, topCategoryClassKey } from '../training-data/label-index.js'

/*
const allTopSleeveTrainingData = longSleeveTrainingData.concat(
  noSleeveTrainingData,
  shortSleeveTrainingData,
  straplessTrainingData
)
const allTopSleeveTrainingLabels = longSleeveDataLabels.concat(
  noSleeveDataLabels,
  shortSleeveDataLabels,
  straplessDataLabels
)
*/

/* List models in Local Storage */
// listModelsInLocalStorage()

/* Remove model from Local Storage */
// removeModelFromLocalStorage('top-sleeve-batched-v4')

const shuffler = (trainingData, desiredLength) => {
  const shuffleMap = tf.util.createShuffledIndices(trainingData.length)
  let dataBatch = []
  trainingData.forEach((image, idx) => {
    dataBatch.push(trainingData[shuffleMap[idx]])
  })
  const shuffledData = dataBatch.slice(0, desiredLength)
  return shuffledData
}

// console.log(
  // blouseTrainingData.length,
  // blouseDataLabels.length,
  // cardiganTrainingData.length,
  // cardiganDataLabels.length,
  // collaredButtonTrainingData.length,
  // collaredButtonDataLabels.length,
  // shirtTrainingData.length,
  // shirtDataLabels.length,
  // sweaterTrainingData.length,
  // sweaterDataLabels.length,
  // sweatshirtTrainingData.length,
  // sweatshirtDataLabels.length,
  // tankTrainingData.length,
  // tankDataLabels.length,
// )

const trainWithBatches = async (
  trainingData,
  trainingLabels,
  numItems,
  labelKey,
  numClasses,
  batchSize,
  localStorageName
) => {
  try {
    const newModel = await trainShuffledBatches(
      trainingData,
      trainingLabels,
      numItems,
      labelKey,
      numClasses,
      batchSize
    )
    console.log('%c Model Completed Training', 'color: #65f2a2; font-weight: bold', newModel)

    newModel.save(`indexeddb://${localStorageName}`)

    console.log(
      `%c Model Saved to Local Storage as ${localStorageName}`,
      'color: #65f2a2; font-weight: bold'
    )
  } catch (err) {
    console.log(err)
  }
}

// trainWithBatches(
//   allTopCategoryTrainingData,
//   allTopCategoryTrainingLabels,
//   5250,
//   topCategoryClassKey,
//   7,
//   5250, // numitems/batchsize = num runs; for example, 140/70 = 2 runs 140/35 = 4 runs
//   'top-category-no-batch'
// )


const shuffleDataBeforeTrain = async (cutoffNum) => {
  const shuffledShirt = shuffler(shirtTrainingData, cutoffNum)
  const shuffledTank = shuffler(tankTrainingData, cutoffNum)
  const shuffledBlouse = shuffler(blouseTrainingData, cutoffNum)
  const shuffledCollaredButton = shuffler(collaredButtonTrainingData, cutoffNum)
  const shuffledSweater = shuffler(sweaterTrainingData, cutoffNum)
  const shuffledSweatshirt = shuffler(sweatshirtTrainingData, cutoffNum)
  const shuffledCardigan = shuffler(cardiganTrainingData, cutoffNum)

  const allTopCategoryTrainingData = shuffledShirt.concat(
    shuffledTank,
    shuffledBlouse,
    shuffledCollaredButton,
    shuffledSweater,
    shuffledSweatshirt,
    shuffledCardigan
  )
  const allTopCategoryTrainingLabels = shirtDataLabels.slice(0,cutoffNum).concat(
    tankDataLabels.slice(0,cutoffNum),
    blouseDataLabels.slice(0,cutoffNum),
    collaredButtonDataLabels.slice(0,cutoffNum),
    sweaterDataLabels.slice(0, cutoffNum),
    sweatshirtDataLabels.slice(0, cutoffNum),
    cardiganDataLabels.slice(0,cutoffNum),
  )

  console.log(allTopCategoryTrainingData, allTopCategoryTrainingLabels)

  await trainWithBatches(
    allTopCategoryTrainingData,
    allTopCategoryTrainingLabels,
    5250,
    topCategoryClassKey,
    7,
    1050, // 5 runs 
    'top-category-750i-1050per-125bs-210e'
  )

}

// shuffleDataBeforeTrain(750)



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

const predictor = async (modelName, labelKey) => {
  for (const testImage in testData) {
    await predict(modelName, testData[testImage], labelKey, testImage)
  }
}

predictor('top-category-750i-1050per-125bs-210e', topCategoryClassKey)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/*
const trainWithoutBatching = async (trainingData, dataLabels, labelKey, numClasses, localStorageName) => {
  try {
    const xs = await getXs(trainingData)
    const ys = await getYs(dataLabels, labelKey)
    const compiledModel = compileModel(numClasses)
    const trainedModel = await fitModel(compiledModel, xs, ys, 0)
    console.log('%c Batch completed successfully', 'color: #4295f4; font-weight: bold')
    trainedModel.save(`indexeddb://${localStorageName}`)
    return trainedModel
  } catch (err) {
    console.log(err)
  }
}

// trainWithoutBatching(allTopSleeveTrainingData, allTopSleeveTrainingLabels, topSleeveClassKey, 4, 'top-sleeve-not-batched')

*/


/*
// Check that your data is right before training! 
console.log(
  'longsleeve data: ',
  longSleeveTrainingData.length,
  '\nlabels: ',
  `"${longSleeveDataLabels[0]}"`,
  longSleeveDataLabels.length
)
console.log(
  'no sleeve data: ',
  noSleeveTrainingData.length,
  '\nlabels: ',
  `"${noSleeveDataLabels[0]}"`,
  noSleeveDataLabels.length
)
console.log(
  'short sleeve data: ',
  shortSleeveTrainingData.length,
  '\nlabels: ',
  `"${shortSleeveDataLabels[0]}"`,
  shortSleeveDataLabels.length
)
console.log(
  'strapless data: ',
  straplessTrainingData.length,
  '\nlabels: ',
  `"${straplessDataLabels[0]}"`,
  straplessDataLabels.length
)
*/
