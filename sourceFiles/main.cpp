#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace dnn;
using namespace std;

vector<string> getOutputNames(const Net& net) {
    static vector<string> names;
    if (names.empty()) {
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<string> layersNames = net.getLayerNames();
        for (int i : outLayers)
            names.push_back(layersNames[i - 1]);
    }
    return names;
}

int main() {
    // Load YOLO
    Net net = readNet("yolo/yolov4.weights", "yolo/yolov4.cfg");
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Load class names
    vector<string> classes;
    ifstream ifs("yolo/coco.names");
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Load image
    Mat imgEarly = imread("images/cat3.jpg");
    Mat img;
	resize(imgEarly, img, Size(1280, 720));
    Mat blob;
    blobFromImage(img, blob, 1 / 255.0, Size(416, 416), Scalar(), true, false);
    net.setInput(blob);

    // Forward pass
    vector<Mat> outs;
    net.forward(outs, getOutputNames(net));

    // Detection
    float confThreshold = 0.5;
    for (auto& output : outs) {
        for (int i = 0; i < output.rows; ++i) {
            float* data = output.ptr<float>(i);
            float confidence = data[4];
            if (confidence > confThreshold) {
                int classId = max_element(data + 5, data + output.cols) - (data + 5);
                float score = data[5 + classId];
                if (score > confThreshold && (classes[classId] == "dog" || classes[classId] == "cat")) {
                    int centerX = (int)(data[0] * img.cols);
                    int centerY = (int)(data[1] * img.rows);
                    int width = (int)(data[2] * img.cols);
                    int height = (int)(data[3] * img.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    rectangle(img, Rect(left, top, width, height), Scalar(255, 0, 0), 2);
                    string label = format("%s: %.2f%%", classes[classId].c_str(), score * 100);
                    int baseline = 0;
                    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
                    top = max(top, labelSize.height);

                    // Draw filled rectangle as background
                    rectangle(img, Point(left, top - labelSize.height - 5),
                        Point(left + labelSize.width, top),
                        Scalar(0, 255, 0), FILLED);

                    // Draw label text over the rectangle
                    putText(img, label, Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
                }
            }
        }
    }

    imshow("Detected", img);
    waitKey(0);
    return 0;
}