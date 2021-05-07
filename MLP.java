import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Scanner;

class MLP {

    public static double[][] weightHidden = new double[20][64];
    public static double[][] weightOutput = new double[10][20];
    public static double[] dataSample = new double[64];
    public static double[] mapArray = new double[10];
    public static double[] outputNeuron = new double[10];
    public static double[] errorOutput = new double[10];
    public static double[] errorHidden = new double[20];
    public static int targetOutput = 0;
    public static double learningRate = 0.01;
    public static double success = 0;
    public static double Accuracy = 0;
    public static double[] sigValues = new double[20];

    // Method made to initialize the range of random numbers
    public static double randNum(double min, double max) {
        //the max and min determine the range
        return (Math.random() * (max - min)) + min;
    }

    //Method that initialize weights
    static void getWeights(double[][] array) {
        for (int i = 0; i < array.length; i++) {
            for (int z = 0; z < array[i].length; z++) {
                // values between -0.3 and 0.3 are being added on to the 2D array
                array[i][z] = randNum(-0.3, 0.3);
            }
        }
    }

    public static void main(String[] args) throws IOException {

        int iterations = 0;

        //Initializing weights
        getWeights(weightOutput);
        getWeights(weightHidden);

        double row = 0;
        double tempsuccess=0;

        while (iterations < 500) {
            //READING TRAINING FILE
            FileReader databaseReader = new FileReader("/Users/ayaanjaleel/Desktop/AI module/NN/CW2SET1.csv");

            //Scanning the file that contains the inputs
            Scanner database = new Scanner(databaseReader);
            row = 0;
            while (database.hasNext()) {

                //reading each row, and storing them in String form
                String str = database.nextLine();
                String[] elements = str.split(",");

                //Converting string inputs to Double
                for (int i = 0; i < 64; i++) {
                    //inputs converted to double gets added onto the dataSample array
                    dataSample[i] = Double.parseDouble(elements[i]);
                }

                //initializing the map array to zeroes
                for (int i = 0; i < mapArray.length; i++) {
                    //all the items in this array are now 0.0
                    mapArray[i] = 0;
                }

                //Mapping the target output in the map array
                targetOutput = Integer.parseInt(elements[64]);
                //if the targetOutput is 6, then the 6th item in the mapArray becomes 1.0
                mapArray[targetOutput] = 1.0;

                //Calling the feed-forward method
                feedforward();

                //This is the test-error method
                //if the desired output matches the actual output
                if (compare(mapArray, outputNeuron)) {
                    //the more the number of matches, the more success
                    success++;
                } else {
                    //if there is no match, this will call the backprop algorithm
                    backprop();
                }

                row++;

            }
            //calculating the accuracy
            Accuracy = (success / row);
            //storing the success value for printing reasons
            tempsuccess=success;
            //reseting success value after each iteration
            success = 0;
            //incrementing each iteration by 1 until it reaches 500
            //System.out.println("Iterations: " + iterations + ", accuracy: " + Accuracy * 100);
            iterations++;
            row++;
        }
        //PRINTING OUT RESULTS
        System.out.println("Result for training:");
        System.out.println("____________________");
        System.out.println("Hidden Neurons: " + weightHidden.length);
        System.out.println("Weights range: (-0.3, 0.3)");

        System.out.println("Learning Rate: 0.01");
        System.out.println("Total Rows: 2810");
        System.out.println("Overall Success: " + tempsuccess);
        System.out.println("Accuracy: % " + Accuracy*100);
        System.out.println(" ");
        System.out.println("===========================");

        //CALLING TESTING METHOD
        testing();
    }

    public static void testing() throws FileNotFoundException {
        //READING TESTING FILE
        FileReader databaseReader = new FileReader("/Users/ayaanjaleel/Desktop/AI module/NN/CW2SET2.csv");

        //reading the file that contains the inputs
        Scanner database = new Scanner(databaseReader);
        double row = 0;
        while (database.hasNext()) {
            //reading each row, and storing them in String form
            String str = database.nextLine();
            String[] elements = str.split(",");


            for (int i = 0; i < 64; i++) {
                //Converting string inputs to Double
                dataSample[i] = Double.parseDouble(elements[i]);
            }

            for (int i = 0; i < mapArray.length; i++) {
                //making the values of the mapArray into 0.0
                mapArray[i] = 0;
            }
            //Mapping the target output
            targetOutput = Integer.parseInt(elements[64]);
            //if the targetOutput is 9, then the 9th item in the mapArray becomes 1.0
            mapArray[targetOutput] = 1.0;

            //Computing the output of the hidden neurons
            feedforward();

            //if the desired output matches the output
            if (compare(mapArray, outputNeuron)) {
                //the more the number of matches, the more the success
                success++;
            }
            row++;
        }

        //calculating the accuracy
        Accuracy = (success / row);

        System.out.println("Result for testing");
        System.out.println("____________________");
        System.out.println("Hidden Neurons: " + weightHidden.length);
        System.out.println("Weights range: (-0.3, 0.3)");
        System.out.println("Iterations: 1" );
        System.out.println("Learning Rate: 0.01");
        System.out.println("Total Rows: 2810");
        System.out.println("Overall Success: " + success);
        System.out.println("Accuracy: % " + Accuracy * 100);
    }

    public static void backprop() {
        //finding the error difference, and storing it in errorOutput
        for (int j = 0; j < 10; j++) {
            errorOutput[j] = mapArray[j] - outputNeuron[j];
        }


        for (int j = 0; j < 20; j++) {
            double errorTemp = 0;
            for (int z = 0; z < 10; z++) {
                //calculating error temp
                errorTemp += errorOutput[z] * weightOutput[z][j];
            }
            errorHidden[j] = (sigValues[j] * (1 - sigValues[j]) * errorTemp);
        }

        //adjusting weights of the output neuron based on the learning rate
        for (int j = 0; j < 10; j++) {
            for (int z = 0; z < 20; z++) {
                weightOutput[j][z] = weightOutput[j][z] + learningRate * sigValues[z] * errorOutput[j];
            }
        }
        //adjusting weights of the hidden neuron based on the learning rate
        for (int j = 0; j < 20; j++) {
            for (int z = 0; z < 64; z++) {
                weightHidden[j][z] = weightHidden[j][z] + learningRate * dataSample[z] * errorHidden[j];
            }
        }
    }

    public static boolean compare(double arr[], double arr2[]) {
        //method used to compare two given arrays
        for (int i = 0; i < arr.length; i++) {
            //used to compare the desired output with the output
            if (arr[i] != arr2[i])
                //if the arrays don't match, it returns false
                return false;
        }
        return true;
    }

    public static void feedforward() {
        double total = 0;
        //Computing the output of the hidden neurons
        for (int i = 0; i < 20; i++) {
            total = 0;
            for (int z = 0; z < 64; z++) {
                //finding the weighted sum
                total += dataSample[z] * weightHidden[i][z];
            }
            //storing the sigmoid values
            sigValues[i] = 1d / (1d + Math.exp(-total));
        }

        //Computing the output of the output neurons
        for (int i = 0; i < 10; i++) {
            double sum = 0;
            for (int z = 0; z < 20; z++) {
                //finding the weighted sum
                sum += sigValues[z] * weightOutput[i][z];
            }
            //the threshold value is 0
            if (sum > 0) {
                //if the sum is more than 0, then the value at i becomes 1.0
                outputNeuron[i] = 1.0;
            } else {
                //if the sum is less than 0, then the value at i becomes 0.0
                outputNeuron[i] = 0.0;
            }
        }
    }
}