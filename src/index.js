import { ConsoleDisplay } from './lib/display'
import { getXs, getYs, loadCustomModel, predictFromTruncated } from './lib/mobileNet'
import { trainModel, predict } from './lib/trainNewModel'
import { trainingData, dataLabels, testData } from './images'
import redBlueModel from '../tfjs-models/red-blue-model.json'


// new ConsoleDisplay().display()

const { blueTest1, redTest1 } = testData

const run = async () => {
  const xs = await getXs(trainingData)
  const ys = await getYs(dataLabels)
  const trainedModel = await trainModel(xs, ys);
  trainedModel.save('indexeddb://red-blue-model')
  // predict(trainedModel, blueTest1)
}

// run();

// List models in Local Storage.
// loadCustomModel('red-blue-model')





