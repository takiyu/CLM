#include "appearance.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

Appearance::Appearance() {}

Appearance::Appearance(const vector<Point2f> init_shape) { init(init_shape); }

void Appearance::init(const vector<Point2f> init_shape) {
    makeTriangleIdxMap(init_shape, this->triangle_map);
}

void Appearance::warp(const Mat& src_image, Mat& dst_image,
                      const vector<Point2f>& src_points,
                      const vector<Point2f>& dst_points) {
    // Obtain triangles and warp
    for (int i = 0; i < this->triangle_map.size(); i++) {
        Point2f src_tri[3];
        Point2f dst_tri[3];
        for (int j = 0; j < 3; j++) {
            src_tri[j] = src_points[this->triangle_map[i][j]];
            dst_tri[j] = dst_points[this->triangle_map[i][j]];
        }
        this->warpTriangle(src_image, dst_image, src_tri, dst_tri);
    }
}

void Appearance::makeTriangleIdxMap(const vector<Point2f>& src_points,
                                    vector<vector<int> >& tri_idxs) {
    tri_idxs.clear();

    Rect rect_container = boundingRect(src_points);
    // Contain boundary points
    rect_container.width += 1;
    rect_container.height += 1;

    // Subdivision
    Subdiv2D subdiv(rect_container);
    subdiv.insert(src_points);

    // Triangles
    vector<Vec6f> tris_6f;
    subdiv.getTriangleList(tris_6f);

    // for each triangle
    for (vector<Vec6f>::iterator it = tris_6f.begin(); it != tris_6f.end();
         it++) {
        cv::Vec6f& vec = *it;

        bool skip_flag = false;
        // temporary triangle indices
        vector<int> tmp_idxs(3);
        // for each vertex
        for (int i = 0; i < 3; i++) {
            Point2f tmp_point(vec[2 * i], vec[2 * i + 1]);
            // Check if points are in the region
            if (rect_container.contains(tmp_point)) {
                // Search same point
                int f;
                for (f = 0; f < src_points.size(); f++) {
                    if (src_points[f] == tmp_point) break;
                }
                tmp_idxs[i] = f;
            }
            // Ignore
            else {
                skip_flag = true;
                break;
            }
        }
        // Register
        if (!skip_flag) tri_idxs.push_back(tmp_idxs);
    }
}

void Appearance::warpTriangle(const Mat& src_image, Mat& dst_image,
                              const Point2f src_tri[],
                              const Point2f dst_tri[]) {
    // 	if(dst_image.empty()) dst_image = src_image.clone();

    // Affine transform
    Mat aff_mat = getAffineTransform(src_tri, dst_tri);
    Mat affed_image;
    warpAffine(src_image, affed_image, aff_mat,
               Size(dst_image.cols, dst_image.rows));

    vector<Point> mask_points(3);
    for (int i = 0; i < 3; i++) {
        mask_points[i] = dst_tri[i];
    }

    // Create Mask
    Mat mask = Mat::zeros(dst_image.rows, dst_image.cols, CV_8UC1);
    fillConvexPoly(mask, mask_points, Scalar(255));

    // Appearancely mask
    affed_image.copyTo(dst_image, mask);

    // 	imshow("dst", dst_image);
    // 	waitKey();
}
