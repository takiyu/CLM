#include "face_data.h"

#include <dirent.h>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

/*===== Muct =====*/
// Read MUCT
void readMUCTLandMarksFile(const string lm_file_name,
                           const string image_dir_name,
                           vector<string>& image_names,
                           vector<vector<Point2f> >& points_vecs) {
    ifstream lm_file(lm_file_name.c_str());
    string line;
    getline(lm_file, line);
    while (!lm_file.eof()) {
        getline(lm_file, line);
        if (line.length() == 0) break;

        size_t p1 = 0, p2;

        //---get image name---
        string ml_name;
        p2 = line.find(",");
        if (p2 == string::npos) {
            cerr << "Invalid MUCT file" << endl;
            exit(0);
        }
        ml_name = line.substr(p1, p2 - p1);
        if (ml_name.compare("i434xe-fn") == 0 || ml_name[1] == 'r') {
            // corrupted data || ignore flipped image
            continue;
        }
        ml_name = image_dir_name + ml_name + ".jpg";

        //---get index---
        string ml_index;
        p1 = p2 + 1;
        p2 = line.find(",", p1);
        if (p2 == string::npos) {
            cerr << "Invalid MUCT file" << endl;
            exit(0);
        }
        ml_index = line.substr(p1, p2 - p1);

        //---get points---
        vector<Point2f> ml_points;
        ml_points.reserve(76);
        for (int i = 0; i < 76; i++) {
            // x
            p1 = p2 + 1;
            p2 = line.find(",", p1);
            if (p2 == string::npos) {
                cerr << "Invalid MUCT file" << endl;
                exit(0);
            }
            string x = line.substr(p1, p2 - p1);
            // y
            p1 = p2 + 1;
            p2 = line.find(",", p1);
            if (p2 == string::npos) {
                // last (no comma)
                if (i == 75) {
                    p2 = line.size() - 1;
                } else {
                    cerr << "Invalid MUCT file" << endl;
                    exit(0);
                }
            }
            string y = line.substr(p1, p2 - p1);
            // add
            ml_points.push_back(Point2f(atoi(x.c_str()), atoi(y.c_str())));
        }

        //---add names and points---
        image_names.push_back(ml_name);
        points_vecs.push_back(ml_points);
    }
    lm_file.close();
}

// Remove shapes containing (0, 0)
void removeIncompleteShape(vector<vector<Point2f> >& points_vecs,
                           vector<string>& image_names) {
    for (int i = 0; i < points_vecs.size(); i++) {
        for (int j = 0; j < points_vecs[i].size(); j++) {
            if (points_vecs[i][j] == Point2f(0, 0)) {
                points_vecs.erase(points_vecs.begin() + i);
                image_names.erase(image_names.begin() + i);
                i--;
                break;
            }
        }
    }
}

// Initialize connections
void initMuctConnections(vector<Vec2i>& connections) {
    connections.clear();
    // outer
    for (int i = 0; i < 14; i++) connections.push_back(Vec2i(i, i + 1));
    // brow
    for (int i = 15; i < 20; i++) connections.push_back(Vec2i(i, i + 1));
    connections.push_back(Vec2i(20, 15));
    for (int i = 21; i < 26; i++) connections.push_back(Vec2i(i, i + 1));
    connections.push_back(Vec2i(26, 21));
    // left eye
    for (int i = 27; i < 30; i++) {
        connections.push_back(Vec2i(i, i + 41));
        connections.push_back(Vec2i(i + 41, i + 1));
    }
    connections.push_back(Vec2i(30, 71));
    connections.push_back(Vec2i(71, 27));
    // right eye
    for (int i = 32; i < 35; i++) {
        connections.push_back(Vec2i(i, i + 40));
        connections.push_back(Vec2i(i + 40, i + 1));
    }
    connections.push_back(Vec2i(35, 75));
    connections.push_back(Vec2i(75, 32));
    // nose
    for (int i = 37; i < 45; i++) connections.push_back(Vec2i(i, i + 1));
    // mouse (outer)
    for (int i = 48; i < 59; i++) connections.push_back(Vec2i(i, i + 1));
    connections.push_back(Vec2i(59, 48));
    // mouse (inner)
    connections.push_back(Vec2i(48, 60));
    for (int i = 60; i < 62; i++) connections.push_back(Vec2i(i, i + 1));
    connections.push_back(Vec2i(54, 62));
    connections.push_back(Vec2i(54, 63));
    for (int i = 63; i < 65; i++) connections.push_back(Vec2i(i, i + 1));
    connections.push_back(Vec2i(48, 65));
}

// Initialize symmetry
void initMuctSymmetry(vector<int>& symmetry) {
    // Initialize to make safty
    symmetry.resize(76);
    for (int i = 0; i < 76; i++) symmetry[i] = i;

    // outer
    for (int i = 0; i <= 14; i++) symmetry[i] = 14 - i;
    // brow
    for (int i = 15; i <= 20; i++) symmetry[i] = i + 6;
    for (int i = 21; i <= 26; i++) symmetry[i] = i - 6;
    // eye 1
    for (int i = 27; i <= 31; i++) symmetry[i] = i + 5;
    for (int i = 32; i <= 36; i++) symmetry[i] = i - 5;
    // nose
    for (int i = 37; i <= 45; i++) symmetry[i] = 82 - i;  // 37+45 - i
    // nostril
    symmetry[46] = 47;
    symmetry[47] = 46;
    // mouse (outer, upper)
    for (int i = 48; i <= 54; i++) symmetry[i] = 102 - i;  // 48+54 - i
    // mouse (outer, lower)
    for (int i = 55; i <= 59; i++) symmetry[i] = 114 - i;  // 55+59 - i
    // mouse (inner)
    symmetry[60] = 62;
    symmetry[62] = 60;
    symmetry[63] = 65;
    symmetry[65] = 63;
    // eye 2
    for (int i = 68; i <= 71; i++) symmetry[i] = i + 4;
    for (int i = 72; i <= 75; i++) symmetry[i] = i - 4;
}

// Extract eye and nose points
void extractEyeAndNosePoints(vector<vector<Point2f> >& points_vecs,
                             vector<int>& symmetry,
                             vector<Vec2i>& connections) {
    // symmetry
    for (int i = symmetry.size() - 1; i >= 0; i--) {
        if (48 <= symmetry[i] && symmetry[i] < 67) {
            symmetry.erase(symmetry.begin() + i);
        }
        // else if(0 <= symmetry[i] && symmetry[i] < 27) {
        //     symmetry.erase(symmetry.begin()+i);
        // }
        else if (67 <= symmetry[i]) {
            symmetry[i] -= (67 - 48);
            symmetry[i] -= (27 - 0);
        }
        // else {
        //     symmetry[i] -= (27 - 0);
        // }
    }
    // connection
    for (int i = connections.size() - 1; i >= 0; i--) {
        for (int j = 0; j < 2; j++) {
            if (48 <= connections[i][j] && connections[i][j] < 67) {
                connections.erase(connections.begin() + i);
                j = 10;  // skip j loop
            }
            // else if(0 <= connections[i][j] && connections[i][j] < 27) {
            //     connections.erase(connections.begin()+i);
            //     j = 100;//skip j loop
            // }
            else if (67 <= connections[i][j]) {
                connections[i][j] -= (67 - 48);
                connections[i][j] -= (27 - 0);
            }
            // else {
            //     connections[i][j] -= (27 - 0);
            // }
        }
    }
    // points
    for (int i = 0; i < points_vecs.size(); i++) {
        points_vecs[i].erase(points_vecs[i].begin() + 48,
                             points_vecs[i].begin() + 67);
        // 		points_vecs[i].erase(points_vecs[i].begin(),
        // points_vecs[i].begin()+27);
    }
}

// Extract important points
void reduceMuctPoints(const vector<Point2f>& src_points,
                      vector<Point2f>& dst_points) {
    dst_points.clear();
    // 	dst_points.reserve(32);
    dst_points.reserve(24);

    dst_points.push_back(src_points[0]);
    dst_points.push_back(src_points[2]);
    dst_points.push_back(src_points[4]);
    // 	dst_points.push_back(src_points[6]);
    dst_points.push_back(src_points[7]);
    // 	dst_points.push_back(src_points[8]);
    dst_points.push_back(src_points[10]);
    dst_points.push_back(src_points[12]);
    dst_points.push_back(src_points[14]);

    dst_points.push_back(src_points[15]);
    dst_points.push_back(src_points[18]);
    dst_points.push_back(src_points[21]);
    dst_points.push_back(src_points[24]);

    dst_points.push_back(src_points[27]);
    // 	dst_points.push_back(src_points[28]);
    dst_points.push_back(src_points[29]);
    // 	dst_points.push_back(src_points[30]);
    dst_points.push_back(src_points[32]);
    // 	dst_points.push_back(src_points[33]);
    dst_points.push_back(src_points[34]);
    // 	dst_points.push_back(src_points[35]);

    dst_points.push_back(src_points[37]);
    // 	dst_points.push_back(src_points[38]);
    dst_points.push_back(src_points[39]);
    dst_points.push_back(src_points[41]);
    dst_points.push_back(src_points[43]);
    // 	dst_points.push_back(src_points[44]);
    dst_points.push_back(src_points[45]);

    dst_points.push_back(src_points[48]);
    dst_points.push_back(src_points[51]);
    dst_points.push_back(src_points[54]);
    dst_points.push_back(src_points[57]);
}

/*===== Helen =====*/
// Get filenames in a directory (must be absolute path)
void getFileNamesInDir(const string& src_dir, vector<string>& dst_names,
                       const int max_count) {
    dst_names.clear();
    // data size counter for the limitation
    int count = 1;

    // Open the directory
    struct dirent* face_dirent;
    DIR* face_dp = opendir(src_dir.c_str());

    while ((face_dirent = readdir(face_dp)) != NULL) {
        // for each face
        if (face_dirent->d_name[0] == '.') continue;  // escape hidden dir

        // output
        stringstream ss;
        ss << src_dir << "/" << face_dirent->d_name;
        dst_names.push_back(ss.str());

        count++;
        if (max_count != 0 && count > max_count) break;
    }
}

// Read Helen
void readHelenFiles(const string& image_dir, const string& point_dir,
                    vector<string>& image_names,
                    vector<vector<Point2f> >& points_vecs) {
    image_names.clear();
    points_vecs.clear();
    // Get file list
    getFileNamesInDir(image_dir, image_names, 0);
    vector<string> point_names;
    getFileNamesInDir(point_dir, point_names, 0);

    assert(image_names.size() == point_names.size());

    // Sort file names
    sort(image_names.begin(), image_names.end());
    sort(point_names.begin(), point_names.end());

    for (int i = 0; i < image_names.size(); i++) {
        // Get the points
        vector<Point2f> points;
        ifstream ifs(point_names[i].c_str());
        if (ifs.fail()) {
            cerr << "Failed to read : " << point_names[i] << endl;
            return;
        }
        string line;
        char sep_char = ' ';
        bool is_body = false;
        int points_num;
        bool end_flag = false;
        // for each line
        while (getline(ifs, line)) {
            stringstream line_ss(line);
            string txt;
            // for each element split by spaces
            while (getline(line_ss, txt, sep_char)) {
                // Get the number of points
                if (txt.compare("n_points:") == 0) {
                    // next
                    getline(line_ss, txt);
                    // prepare for output
                    points.clear();
                    points_num = atoi(txt.c_str());
                    points.reserve(points_num);
                }
                // Start body with `{`
                else if (txt.compare("{") == 0)
                    is_body = true;
                // End body with '}'
                else if (txt.compare("}") == 0) {
                    if (points_num != points.size()) {
                        cout << points_num << endl;
                        cout << points.size() << endl;
                        cerr << "Invalid data : " << point_names[i] << endl;
                    }
                    ifs.close();
                    end_flag = true;
                    break;
                }
                // Read the body
                else if (is_body) {
                    float x, y;
                    // string to float
                    stringstream stof_ss;
                    stof_ss << txt;
                    stof_ss >> x;
                    stof_ss.clear();
                    // y coordinate, too
                    getline(line_ss, txt);
                    stof_ss << txt;
                    stof_ss >> y;
                    // Register
                    points.push_back(Point2f(x, y));
                }
            }
            if (end_flag) break;
        }

        points_vecs.push_back(points);
    }
}

// Initialize connections
void initHelenConnections(vector<Vec2i>& connections) {
    connections.clear();
    // outer
    for (int i = 0; i < 16; i++) connections.push_back(Vec2i(i, i + 1));
    // brow
    for (int i = 17; i < 21; i++) connections.push_back(Vec2i(i, i + 1));
    for (int i = 22; i < 26; i++) connections.push_back(Vec2i(i, i + 1));
    // nose
    for (int i = 27; i < 30; i++) connections.push_back(Vec2i(i, i + 1));
    // nose (lower)
    for (int i = 31; i < 35; i++) connections.push_back(Vec2i(i, i + 1));
    // left eye
    for (int i = 36; i < 41; i++) connections.push_back(Vec2i(i, i + 1));
    connections.push_back(Vec2i(36, 41));
    // right eye
    for (int i = 42; i < 47; i++) connections.push_back(Vec2i(i, i + 1));
    connections.push_back(Vec2i(42, 47));

    // mouse (outer)
    for (int i = 48; i < 59; i++) connections.push_back(Vec2i(i, i + 1));
    connections.push_back(Vec2i(48, 59));
    // mosue (inner)
    for (int i = 60; i < 67; i++) connections.push_back(Vec2i(i, i + 1));
    connections.push_back(Vec2i(60, 67));
}

// Initialize symmetry
void initHelenSymmetry(vector<int>& symmetry) {
    // Initialize to make safty
    symmetry.resize(68);
    for (int i = 0; i < 68; i++) symmetry[i] = i;

    // outer
    for (int i = 0; i <= 16; i++) symmetry[i] = 16 - i;
    // brow
    for (int i = 17; i <= 26; i++) symmetry[i] = 43 - i;
    // nose (lower)
    for (int i = 31; i <= 35; i++) symmetry[i] = 66 - i;

    // eye 1
    for (int i = 36; i <= 39; i++) symmetry[i] = 81 - i;
    for (int i = 42; i <= 45; i++) symmetry[i] = 81 - i;
    // eye 2
    for (int i = 40; i <= 41; i++) symmetry[i] = 87 - i;
    for (int i = 46; i <= 47; i++) symmetry[i] = 87 - i;

    // mouse (outer)
    for (int i = 48; i <= 54; i++) symmetry[i] = 102 - i;
    for (int i = 55; i <= 59; i++) symmetry[i] = 114 - i;
    // mouse (inner)
    for (int i = 60; i <= 64; i++) symmetry[i] = 124 - i;
    for (int i = 65; i <= 67; i++) symmetry[i] = 132 - i;
}

// Extract important points
void reduceHelenPoints(const vector<Point2f>& src_points,
                       vector<Point2f>& dst_points) {
    dst_points.clear();
    for (int i = 0; i < src_points.size(); i++) {
        if (i == 0 || i == 16) continue;
        dst_points.push_back(src_points[i]);
    }
}

void reduceHelenPoints2(const vector<Point2f>& src_points,
                        vector<Point2f>& dst_points) {
    dst_points.clear();

    // outer
    // 	dst_points.push_back(src_points[0]);
    dst_points.push_back(src_points[1]);
    dst_points.push_back(src_points[2]);
    dst_points.push_back(src_points[4]);
    dst_points.push_back(src_points[6]);
    dst_points.push_back(src_points[7]);
    dst_points.push_back(src_points[8]);
    dst_points.push_back(src_points[10]);
    dst_points.push_back(src_points[12]);
    dst_points.push_back(src_points[14]);
    dst_points.push_back(src_points[15]);
    // 	dst_points.push_back(src_points[16]);
    // left brow
    dst_points.push_back(src_points[17]);
    dst_points.push_back(src_points[19]);
    dst_points.push_back(src_points[21]);
    // right brow
    dst_points.push_back(src_points[22]);
    dst_points.push_back(src_points[24]);
    dst_points.push_back(src_points[26]);
    // nose (center)
    dst_points.push_back(src_points[27]);
    dst_points.push_back(src_points[30]);
    // nose (lower)
    dst_points.push_back(src_points[31]);
    dst_points.push_back(src_points[35]);
    // left eye
    dst_points.push_back(src_points[36]);
    dst_points.push_back(src_points[38]);  //
    dst_points.push_back(src_points[39]);
    dst_points.push_back(src_points[40]);  //
    // right eye
    dst_points.push_back(src_points[42]);
    dst_points.push_back(src_points[43]);  //
    dst_points.push_back(src_points[45]);
    dst_points.push_back(src_points[47]);  //
    // mouse (outer)
    dst_points.push_back(src_points[48]);
    dst_points.push_back(src_points[50]);  // left
    dst_points.push_back(src_points[51]);
    dst_points.push_back(src_points[52]);  // right
    dst_points.push_back(src_points[54]);
    dst_points.push_back(src_points[55]);
    dst_points.push_back(src_points[57]);
    dst_points.push_back(src_points[59]);
    // mouse (inner)
    dst_points.push_back(src_points[60]);
    dst_points.push_back(src_points[62]);
    dst_points.push_back(src_points[64]);
    dst_points.push_back(src_points[66]);
}
