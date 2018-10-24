import { ConsoleDisplay } from './lib/display'
import { getXs, oneHot, getYs } from './lib/mobileNet'
import { trainModel, predict } from './lib/trainNewModel'
import { trainingData, dataLabels, testData } from './images'


// new ConsoleDisplay().display()

// formatImage(testImage)

// predictFromTruncated(testImage)

const run = async () => {
  const xs = await getXs(trainingData)
  const ys = await getYs(dataLabels)
  const trainedModel = await trainModel(xs, ys);
  await predict(trainedModel, testData)
}
// getXs(trainingData)

// run();

// oneHot(3,5).print()

// getYs(dataLabels)
// const ys = getYs(dataLabels)
// console.log

