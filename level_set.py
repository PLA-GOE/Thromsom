import copy

import itk as itk
import argparse
from PIL import Image, ImageTk
import numpy as np


def show_mono(image_in):
    image_min = np.amin(image_in)
    show_image = (((image_in - image_min) / (np.amax(image_in) - image_min)) * 255.0).astype(np.uint8)
    img = Image.fromarray(show_image, 'L')
    img.show()

def level_set(image_in, seed_x, seed_y):
    x = seed_x
    y = seed_y

    initial_distance = 5.0  # 5.0

    sigma = 1.5 # 1.5 #Wanddicke
    sigmoid_alpha = -0.05  # -0.05 #Invertfaktor
    sigmoid_beta = 6  # 6 #Wand"Feinheit"

    propagation_scaling = 10.0  # 10.0
    number_of_iterations = 10000

    output_image = "out.png"

    show_im = copy.deepcopy(image_in)
    show_im[seed_x - 3:seed_x + 2, seed_y - 3:seed_y + 2] = 255
    show_mono(show_im)

    seedValue = -initial_distance

    Dimension = 2

    InputPixelType = itk.F
    OutputPixelType = itk.UC

    InputImageType = itk.Image[InputPixelType, Dimension]
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    WriterType = itk.ImageFileWriter[OutputImageType]
    image_in = itk.image_from_array(image_in, ttype=(InputImageType,))
    # image_in = itk.PyBuffer[InputImageType].GetImageFromArray(image_in)
    print(image_in)
    SmoothingFilterType = itk.CurvatureAnisotropicDiffusionImageFilter[
        InputImageType, InputImageType
    ]
    smoothing = SmoothingFilterType.New()
    smoothing.SetTimeStep(0.125)
    smoothing.SetNumberOfIterations(5)
    smoothing.SetConductanceParameter(9.0)
    smoothing.SetInput(image_in)

    GradientFilterType = itk.GradientMagnitudeRecursiveGaussianImageFilter[
        InputImageType, InputImageType
    ]
    gradientMagnitude = GradientFilterType.New()
    gradientMagnitude.SetSigma(sigma)
    gradientMagnitude.SetInput(image_in)

    SigmoidFilterType = itk.SigmoidImageFilter[InputImageType, InputImageType]
    sigmoid = SigmoidFilterType.New()
    sigmoid.SetOutputMinimum(0.0)
    sigmoid.SetOutputMaximum(1.0)
    sigmoid.SetAlpha(sigmoid_alpha)
    sigmoid.SetBeta(sigmoid_beta)
    sigmoid.SetInput(gradientMagnitude.GetOutput())

    FastMarchingFilterType = itk.FastMarchingImageFilter[InputImageType, InputImageType]
    fastMarching = FastMarchingFilterType.New()

    GeoActiveContourFilterType = itk.GeodesicActiveContourLevelSetImageFilter[
        InputImageType, InputImageType, InputPixelType
    ]
    geodesicActiveContour = GeoActiveContourFilterType.New()
    geodesicActiveContour.SetPropagationScaling(propagation_scaling)
    geodesicActiveContour.SetCurvatureScaling(1.0)
    geodesicActiveContour.SetAdvectionScaling(1.0)
    geodesicActiveContour.SetMaximumRMSError(0.02)
    geodesicActiveContour.SetNumberOfIterations(number_of_iterations)
    geodesicActiveContour.SetInput(fastMarching.GetOutput())
    geodesicActiveContour.SetFeatureImage(sigmoid.GetOutput())

    ThresholdingFilterType = itk.BinaryThresholdImageFilter[InputImageType, OutputImageType]
    thresholder = ThresholdingFilterType.New()
    thresholder.SetLowerThreshold(-1000.0)
    thresholder.SetUpperThreshold(0.0)
    thresholder.SetOutsideValue(itk.NumericTraits[OutputPixelType].min())
    thresholder.SetInsideValue(itk.NumericTraits[OutputPixelType].max())
    thresholder.SetInput(geodesicActiveContour.GetOutput())

    print("GOUT:",geodesicActiveContour.GetOutput())

    seedPosition = itk.Index[Dimension]()
    seedPosition[1] = x
    seedPosition[0] = y

    node = itk.LevelSetNode[InputPixelType, Dimension]()
    node.SetValue(seedValue)
    node.SetIndex(seedPosition)

    seeds = itk.VectorContainer[itk.UI, itk.LevelSetNode[InputPixelType, Dimension]].New()
    seeds.Initialize()
    seeds.InsertElement(0, node)

    fastMarching.SetTrialPoints(seeds)
    fastMarching.SetSpeedConstant(1.0)

    CastFilterType = itk.RescaleIntensityImageFilter[InputImageType, OutputImageType]

    caster1 = CastFilterType.New()
    caster2 = CastFilterType.New()
    caster3 = CastFilterType.New()
    caster4 = CastFilterType.New()

    writer1 = WriterType.New()
    writer2 = WriterType.New()
    writer3 = WriterType.New()
    writer4 = WriterType.New()

    caster1.SetInput(smoothing.GetOutput())
    writer1.SetInput(caster1.GetOutput())
    writer1.SetFileName("GeodesicActiveContourImageFilterOutput1.png")
    caster1.SetOutputMinimum(itk.NumericTraits[OutputPixelType].min())
    caster1.SetOutputMaximum(itk.NumericTraits[OutputPixelType].max())
    writer1.Update()

    caster2.SetInput(gradientMagnitude.GetOutput())
    writer2.SetInput(caster2.GetOutput())
    writer2.SetFileName("GeodesicActiveContourImageFilterOutput2.png")
    caster2.SetOutputMinimum(itk.NumericTraits[OutputPixelType].min())
    caster2.SetOutputMaximum(itk.NumericTraits[OutputPixelType].max())
    writer2.Update()

    caster3.SetInput(sigmoid.GetOutput())
    writer3.SetInput(caster3.GetOutput())
    writer3.SetFileName("GeodesicActiveContourImageFilterOutput3.png")
    caster3.SetOutputMinimum(itk.NumericTraits[OutputPixelType].min())
    caster3.SetOutputMaximum(itk.NumericTraits[OutputPixelType].max())
    writer3.Update()

    caster4.SetInput(fastMarching.GetOutput())
    writer4.SetInput(caster4.GetOutput())
    writer4.SetFileName("GeodesicActiveContourImageFilterOutput4.png")
    caster4.SetOutputMinimum(itk.NumericTraits[OutputPixelType].min())
    caster4.SetOutputMaximum(itk.NumericTraits[OutputPixelType].max())

    fastMarching.SetOutputSize((image_in.shape[1],image_in.shape[0]))

    print(thresholder.GetOutput())
    writer = WriterType.New()
    writer.SetFileName(output_image)
    writer.SetInput(thresholder.GetOutput())
    writer.Update()

    print(
        "Max. no. iterations: " + str(geodesicActiveContour.GetNumberOfIterations()) + "\n"
    )
    print("Max. RMS error: " + str(geodesicActiveContour.GetMaximumRMSError()) + "\n")
    print(
        "No. elpased iterations: "
        + str(geodesicActiveContour.GetElapsedIterations())
        + "\n"
    )
    print("RMS change: " + str(geodesicActiveContour.GetRMSChange()) + "\n")

    writer4.Update()

    InternalWriterType = itk.ImageFileWriter[InputImageType]

    mapWriter = InternalWriterType.New()
    mapWriter.SetInput(fastMarching.GetOutput())
    mapWriter.SetFileName("GeodesicActiveContourImageFilterOutput4.mha")
    mapWriter.Update()

    speedWriter = InternalWriterType.New()
    speedWriter.SetInput(sigmoid.GetOutput())
    speedWriter.SetFileName("GeodesicActiveContourImageFilterOutput3.mha")
    speedWriter.Update()

    gradientWriter = InternalWriterType.New()
    gradientWriter.SetInput(gradientMagnitude.GetOutput())
    gradientWriter.SetFileName("GeodesicActiveContourImageFilterOutput2.mha")
    gradientWriter.Update()

    return thresholder.GetOutput()
