#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

/*===== Muct Dataset =====*/
// Read MUCT
void readMUCTLandMarksFile(const std::string lm_file_name,
                           const std::string image_dir_name,
                           std::vector<std::string>& image_names,
                           std::vector<std::vector<cv::Point2f> >& points_vecs);
// Remove shapes containing (0, 0)
void removeIncompleteShape(std::vector<std::vector<cv::Point2f> >& points_vecs,
                           std::vector<std::string>& image_names);
// Initialize connections
void initMuctConnections(std::vector<cv::Vec2i>& connections);
// Initialize symmetry
void initMuctSymmetry(std::vector<int>& symmetry);
// Extract eye and nose points
void extractEyeAndNosePoints(
    std::vector<std::vector<cv::Point2f> >& points_vecs,
    std::vector<int>& symmetry, std::vector<cv::Vec2i>& connections);
// Extract important points
void reduceMuctPoints(const std::vector<cv::Point2f>& src_points,
                      std::vector<cv::Point2f>& dst_points);

/*===== Helen Dataset =====*/
// Get filenames in a directory (must be absolute path)
void getFileNamesInDir(const std::string& src_dir,
                       std::vector<std::string>& dst_names,
                       const int max_count);
// Read Helen (Broken now)
// void readHelenFiles(const std::string& image_dir, const std::string&
// point_dir,
//                     std::vector<std::string>& image_names,
//                     std::vector<std::vector<cv::Point2f> >& points_vecs);
// Initialize connections
void initHelenConnections(std::vector<cv::Vec2i>& connections);
// Initialize symmetry
void initHelenSymmetry(std::vector<int>& symmetry);
// Extract important points
void reduceHelenPoints(const std::vector<cv::Point2f>& src_points,
                       std::vector<cv::Point2f>& dst_points);
void reduceHelenPoints2(const std::vector<cv::Point2f>& src_points,
                        std::vector<cv::Point2f>& dst_points);
