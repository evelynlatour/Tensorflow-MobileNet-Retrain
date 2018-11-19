import { predictFromTruncated } from './lib/mobileNet'
import { getXs, getYs } from './lib/createXYs'
import { trainShuffledBatches, compileModel, fitModel } from './lib/trainNewModel'

import {
  predict,
  listModelsInLocalStorage,
  loadCustomModel,
  removeModelFromLocalStorage,
} from './lib/manageLocalModels'

import {
  longSleeveTrainingData,
  longSleeveDataLabels,
  noSleeveTrainingData,
  noSleeveDataLabels,
  shortSleeveTrainingData,
  shortSleeveDataLabels,
  straplessTrainingData,
  straplessDataLabels,
} from '../training-data'

import { testData } from '../test-data'

import { topSleeveClassKey } from '../training-data/label-index.js'

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

/* List models in Local Storage */
// listModelsInLocalStorage()

/* Remove model from Local Storage */
// removeModelFromLocalStorage('...')

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
//   allTopSleeveTrainingData,
//   allTopSleeveTrainingLabels,
//   1200,
//   topSleeveClassKey,
//   4,
//   1200, // numitems/batchsize = num runs; for example, 140/70 = 2 runs 140/35 = 4 runs
//   'top-sleeve-batched-v4.3'
// )

const predictor = async (modelName, labelKey) => {
  for (const testImage in testData) {
    await predict(modelName, testData[testImage], labelKey, testImage)
  }
}

// predictor('top-sleeve-batched-v4.3', topSleeveClassKey)

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
