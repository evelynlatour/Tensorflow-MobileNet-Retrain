import {
  getXs,
  getYs,
  loadCustomModel,
  listModelsInLocalStorage,
  predictFromTruncated,
} from './lib/mobileNet'
import { predict, trainShuffledBatches } from './lib/trainNewModel'
import { trainingData, dataLabels, testData } from './images'

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

import { topSleeveClassKey } from '../training-data/label-index.js'

/* List models in Local Storage */
// listModelsInLocalStorage()

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

const trainAndSaveNewModel = async (
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

trainAndSaveNewModel(
  allTopSleeveTrainingData,
  allTopSleeveTrainingLabels,
  130,
  topSleeveClassKey,
  4,
  26,
  'top-sleeve'
)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

const { blueTest1, redTest1 } = testData

const run = async () => {
  const xs = await getXs(trainingData)
  const ys = await getYs(dataLabels)
  const trainedModel = await trainModel(xs, ys)
  // trainedModel.save('indexeddb://red-blue-model')
  // predict(trainedModel, blueTest1)
}

// const trainBatch = async (model, labelKey, trainingData, dataLabels) => {
//   try {
//     const xs = await getXs(trainingData)
//     const ys = await getYs(dataLabels, labelKey)
//     const trainedModel = await fitModel(model, xs, ys)
//     console.log('%c Batch completed successfully', 'color: #4295f4; font-weight: bold')
//     return trainedModel
//   } catch (err) {
//     console.log(err)
//   }
// }
