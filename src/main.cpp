#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "drawLandmarks.hpp"


using namespace std;
using namespace cv;
using namespace cv::face;

#include <iostream>

int main(int argc,char** argv)
{
    CommandLineParser parser(argc, argv,
        "{help h||}"
        "{face_cascade|D:/Repositories/opencv_sandbox/build/installed/Windows/opencv/etc/haarcascades/haarcascade_frontalface_alt2.xml|Path to face cascade.}"
        "{camera|0|Camera device number.}");
    parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
        "You can use Haar or LBP features.\n\n");
    parser.printMessage();

    String face_cascade_name = samples::findFile(parser.get<String>("face_cascade"));

    CascadeClassifier faceDetector;

    //-- 1. Load the cascades
    if (!faceDetector.load(face_cascade_name))
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };


    // Create an instance of Facemark
    Ptr<Facemark> facemark = FacemarkLBF::create();

    // Load landmark detector
    facemark->loadModel("D:/Repositories/opencv_sandbox/src/lbfmodel.yaml");

    // Set up webcam for video capture
    int camera_device = parser.get<int>("camera");
    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open(camera_device);
    if (!capture.isOpened())
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
    
    // Variable to store a video frame and its grayscale 
    Mat frame, gray;
    
    // Read a frame
    while(capture.read(frame))
    {      
      // Find face
      vector<Rect> faces;
      // Convert frame to grayscale because
      // faceDetector requires grayscale image.
      cvtColor(frame, gray, COLOR_BGR2GRAY);

      // Detect faces
      faceDetector.detectMultiScale(gray, faces);
      
      // Variable for landmarks. 
      // Landmarks for one face is a vector of points
      // There can be more than one face in the image. Hence, we 
      // use a vector of vector of points. 
      vector< vector<Point2f> > landmarks;
      
      // Run landmark detector
      bool success = facemark->fit(frame,faces,landmarks);
      
      if(success)
      {
        // If successful, render the landmarks on the face
        for(int i = 0; i < landmarks.size(); i++)
        {
          drawLandmarks(frame, landmarks[i]);
        }
      }

      // Display results 
      imshow("Facial Landmark Detection", frame);
      // Exit loop if ESC is pressed
      if (waitKey(1) == 27) break;
      
    }
    //std::cout << "Hi" << std::endl;
    
    return 0;
}
