import {
  getXs,
  getYs,
  predictFromTruncated,
} from './lib/mobileNet'
import { predict, trainShuffledBatches, compileModel, fitModel, listModelsInLocalStorage, loadCustomModel, removeModelFromLocalStorage
} from './lib/trainNewModel'
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

import {  
  noSleeveTest1,
  noSleeveTest2,
  longSleeveTest1,
  longSleeveTest2,
  shortSleeveTest1,
  shortSleeveTest2, 
} from '../test-data'

import { topSleeveClassKey } from '../training-data/label-index.js'

/* List models in Local Storage */
// listModelsInLocalStorage()

/* Remove model from Local Storage */
// removeModelFromLocalStorage('...')


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
//   140,
//   topSleeveClassKey,
//   4,
//   70, // 140/70 = 2 runs
//   'top-sleeve-batched2'
// )



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//



// predict('top-sleeve-batched2', noSleeveTest1, topSleeveClassKey, 'no sleeve')
// predict('top-sleeve-batched2', noSleeveTest2, topSleeveClassKey, 'no sleeve')
// predict('top-sleeve-batched2', shortSleeveTest1, topSleeveClassKey, 'short sleeve')
// predict('top-sleeve-batched2', shortSleeveTest2, topSleeveClassKey, 'short sleeve')
// predict('top-sleeve-batched2', longSleeveTest1, topSleeveClassKey, 'long sleeve')
// predict('top-sleeve-batched2', longSleeveTest2, topSleeveClassKey, 'long sleeve')


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