#include "Reference_Image.h"

// Creates a layer of an image
void PrepareReferenceImage(ImageInfo srcImage, string outputDir, int numOfTestedTools){
    // Calculate once
    int numElements = srcImage.width * srcImage.height;

    // Open output file
    ofstream outFile;
    outFile.open(outputDir);

    // Iterate all color channels
    for(int colorChannel = 0; colorChannel < 3; colorChannel++){
        // Iterate all pixels
        for(int index = 0; index < numElements; index++){
            // Writes to file layer pixel by pixel
            outFile << static_cast<float>(srcImage.frame->data[colorChannel][index]) / static_cast<float>(numOfTestedTools) << endl;
        }

        // Writes new line
        outFile << "---" << endl;
    }

    // Close output file
    outFile.close();
}

// Creates an image from layers
void CreateReferenceImage(ImageInfo referenceImage, vector<string> inputDirs){
    // Calculate once
    int numElements = referenceImage.width * referenceImage.height;

    // Open input files
    vector<ifstream*> inFiles = vector<ifstream*>();
    for(int index = 0; index < inputDirs.size(); index++){
        // Temporary variable for data stream
        ifstream* tempInFile = new ifstream();
        // Open input file
        tempInFile->open(inputDirs.at(index));
        // Save input stream in a vector
        if(tempInFile->is_open())
            inFiles.push_back(tempInFile);
    }

    // Temporary float image
    vector<vector<float>> floatImage = vector<vector<float>>();
    // For each color channel
    for(int colorChannel = 0; colorChannel < 3; colorChannel++){
        // Temporary channel
        vector<float> floatChannel = vector<float>();
        // For each point of the channel
        for(int index = 0; index < numElements; index++){
            // Assign 0.0f value to each point
            floatChannel.push_back(0.f);
        }

        // Adds float channel to float image
        floatImage.push_back(floatChannel);
    }

    // Temporary values
    string line;
    int colorChannel = 0;
    int index;

    // Iterate all input files
    for(int inFileIndex = 0; inFileIndex < inFiles.size(); inFileIndex++){
        // Process file
        while((*inFiles[inFileIndex]).good()){
            // Read line from file
            getline(*inFiles[inFileIndex], line);

            // Verify if it is end of the file
            if(colorChannel >= 3){
                index = 0;
                colorChannel = 0;
                break;
            }

            // Verify if it is line break
            if(line == "---"){
                index = 0;
                colorChannel++;
                continue;
            }

            // Adds value of layer to float image
            floatImage[colorChannel][index] = floatImage[colorChannel][index] + strtof(line.c_str(), NULL);

            // Increments
            index++;
        }
    }

    // Close all files
    for(int inFileIndex = 0; inFileIndex < inFiles.size(); inFileIndex++)
        (*inFiles.at(inFileIndex)).close();

    // For each color channel
    for(int colorChannel = 0; colorChannel < 3; colorChannel++){
        // For each pixel
        for(int index = 0; index < numElements; index++){
            // Rounds float to uint8_t
            referenceImage.frame->data[colorChannel][index] = float2uint8_t(floatImage.at(colorChannel).at(index));
        }
    }
}