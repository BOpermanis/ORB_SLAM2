home_dir=$(pwd)
#echo "Configuring and building Thirdparty/DBoW2 ..."
#
#cd Thirdparty/DBoW2
#mkdir build
#cd build
#cmake .. -DCMAKE_BUILD_TYPE=Release
#make -j
#
#cd ../../g2o
#
#echo "Configuring and building Thirdparty/g2o ..."
#
#mkdir build
#cd build
#cmake .. -DCMAKE_BUILD_TYPE=Release
#make -j
#
#cd ../../../
#
#echo "Uncompress vocabulary ..."
#
#cd Vocabulary
#tar -xf ORBvoc.txt.tar.gz
#cd ..
#
#echo "Configuring and building ORB_SLAM2 ..."

#rm -rf build
# building pcl
cd /
apt install -y libproj-dev
git clone https://github.com/BOpermanis/SP-SLAM.git
wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.8.0.tar.gz
tar -xf pcl-1.8.0.tar.gz
cd pcl-pcl-1.8.0 && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2 install

# building orbslam
cd $home_dir
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
