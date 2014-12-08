#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <dirent.h>

using namespace cv;
using namespace std;

/*===== Muct =====*/
//MUCTのLandMarksファイルを読み取り
void readMUCTLandMarksFile(const string lm_file_name,
						   const string image_dir_name,
						   vector<string>& image_names,
						   vector<vector<Point2f> >& points_vecs);
//MUCTの(0,0)を含む形状を削除
void removeIncompleteShape(vector< vector<Point2f> >& points_vecs,
						   vector<string>& image_names);
//Muctのconnectionsの初期化
void initMuctConnections(vector<Vec2i>& connections);
//Muctのsymmetryの初期化
void initMuctSymmetry(vector<int>& symmetry);
//Muctデータ群から目と鼻のみを抽出
void extractEyeAndNosePoints(vector<vector<Point2f> >& points_vecs,
							 vector<int>& symmetry,
							 vector<Vec2i>& connections);
//Muctの中間点などを削除
void reduceMuctPoints(const vector<Point2f>& src_points,
					  vector<Point2f>& dst_points);

/*===== Helen =====*/
//dir内のファイル名を取得 (max_countで取得最大量を指定,0で全て)(絶対パス指定)
void getFileNamesInDir(const string& src_dir,
					   vector<string>& dst_names,
					   const int max_count);
//Helen学習データを読み取り
void readHelenFiles(const string& image_dir,
					const string& point_dir,
					vector<string>& image_names,
					vector<vector<Point2f> >& points_vecs);
//Helenのconnectionsの初期化
void initHelenConnections(vector<Vec2i>& connections);
//Helenのsymmetryの初期化
void initHelenSymmetry(vector<int>& symmetry);
//Helenの中間点などを削除
void reduceHelenPoints(const vector<Point2f>& src_points,
					   vector<Point2f>& dst_points);
void reduceHelenPoints2(const vector<Point2f>& src_points,
						vector<Point2f>& dst_points);
