#pragma once

#include <iostream>
#include <thread>
#include <mutex>
#include <future>
#include <condition_variable>
#include <unistd.h>
#include <sstream>
#include <string>


#include "open3d/Open3D.h"
#include <open3d/io/ImageIO.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/geometry/VoxelGrid.h>
#include <open3d/geometry/Image.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/geometry/RGBDImage.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/pipelines/registration/Registration.h>


#if defined(MC12705)
#include "mc12705load.h"
#elif defined(NM_CARD)
#include "nm_card_load.h"
#elif defined(NM_MEZZO)
#include "nm_mezzo_load.h"
#elif defined(NM_QUAD)
#include "nm_quad_load.h"
#elif defined(NEURO_MEZZANINE)
#include "neuro_mezzanine_load.h"
#else
#error "Unknown board!"
#endif

extern pthread_mutex_t work_mutex; //PTHREAD_MUTEX_INITIALIZER;
extern pthread_cond_t  wait_cond;//  = PTHREAD_COND_INITIALIZER;

const int num_threads = 2;

extern int pipe_fd[2];
extern PL_CoreNo all_core[num_threads];
extern std::promise<int> test_promise[num_threads];
extern PL_Board* board;
extern int in_progress;
extern int do_wait;

extern const char* NM_PART_FILE_NAME;// = "./nmc/nmc_part.abs";
extern const char* CCPU_PART_FILE_NAME; // = "./arm/central_arm_part.elf";
extern const char* PC_PART_FILE_NAME; // = "./arm/cluster_arm_part.elf";

#undef USE_CHAN_EDCL
#ifdef USE_CHAN_EDCL
unsigned char HOST_MAC_ADDRESS[] = {0x00, 0x14, 0xD1, 0x16, 0x69, 0xD0};
unsigned char BOARD_MAC_ADDRESS[] = {0xEC, 0x17, 0x66, 0x64, 0x08, 0x00};
#endif

struct Pcd
{
	std::vector<Eigen::Vector3d> points;
	std::vector<Eigen::Vector3d> normals;
};

struct Geometry
{
    Pcd* source;
	Pcd* target;
	std::vector<Eigen::Vector2i>* corres;
};

struct Buff
{
	PL_Word* buff;
	size_t size;
};

struct SerializedData
{
	Buff sourcePoints;
	Buff sourceNormals;
	Buff targetPoints;
	Buff targetNormals;
	Buff corres;
};

struct MatrixData
{
    Eigen::Matrix<double, 6, 6> matrix;
    Eigen::Matrix<double, 6, 1> vector;
    double r2_sum;
};

void calc_pcd(
    int index, int start, int end,
    const SerializedData& data,
    std::promise<int>& _test_promise);

class PointCloudRegistration
{
private:
    Geometry geometry;
    SerializedData serializedData;
    MatrixData result;

public:
    PointCloudRegistration(
		const open3d::geometry::PointCloud& source,
		const open3d::geometry::PointCloud& target)
	{
		convertToGeom(source, target);
		serializeData();
	}

    std::pair<PL_Word*, size_t> convertVectorToBuffer(const std::vector<Eigen::Vector3d>& points);

    std::pair<PL_Word*, size_t> convertVectorToBuffer(const std::vector<Eigen::Vector2i>& points);

    void serializeData();

	void convertToGeom(
		const open3d::geometry::PointCloud& source,
		const open3d::geometry::PointCloud& target);

	// Implementation of ICP registration using geometry data
	int performICPRegistration();
    MatrixData getResult() const;
};
