#include "PcdRegistration.hpp"


std::pair<PL_Word*, size_t> PointCloudRegistration::convertVectorToBuffer(const std::vector<Eigen::Vector3d>& points) {
    // Determine the size of the buffer
    size_t buffer_size = points.size() * 3;

    // Create the buffer
    PL_Word* buffer = new PL_Word[buffer_size];

    // Flatten Eigen::Vector3d to PL_Word buffer
    for (size_t i = 0; i < points.size(); ++i) {
        buffer[i * 3]     = static_cast<PL_Word>(points[i](0));
        buffer[i * 3 + 1] = static_cast<PL_Word>(points[i](1));
        buffer[i * 3 + 2] = static_cast<PL_Word>(points[i](2));
    }

    return {buffer, buffer_size};
}

std::pair<PL_Word*, size_t> PointCloudRegistration::convertVectorToBuffer(const std::vector<Eigen::Vector2i>& points) {
    // Determine the size of the buffer
    size_t buffer_size = points.size() * 2;

    // Create the buffer
    PL_Word* buffer = new PL_Word[buffer_size];

    // Flatten Eigen::Vector2i to PL_Word buffer
    for (size_t i = 0; i < points.size(); ++i) {
        buffer[i * 2]     = static_cast<PL_Word>(points[i](0));
        buffer[i * 2 + 1] = static_cast<PL_Word>(points[i](1));
    }

    return {buffer, buffer_size};
}

void PointCloudRegistration::serializeData() {
    // Source
    auto [sourcePoints, sourcePointsSize] = convertVectorToBuffer(geometry.source->points);
    std::cout << "source points buff size: " << sourcePointsSize << std::endl;

    auto [sourceNormals, sourceNormalsSize] = convertVectorToBuffer(geometry.source->normals);
    std::cout << "source normals buff size: " << sourceNormalsSize << std::endl;

    // Target

    auto [targetPoints, targetPointsSize] = convertVectorToBuffer(geometry.target->points);
    std::cout << "target points buff size: " << targetPointsSize << std::endl;

    auto [targetNormals, targetNormalsSize] = convertVectorToBuffer(geometry.target->normals);
    std::cout << "target normals buff size: " << targetNormalsSize << std::endl;

    auto [corres, corresSize] = convertVectorToBuffer(*geometry.corres);

    serializedData.sourcePoints = {sourcePoints, sourcePointsSize};
    serializedData.sourceNormals = {sourceNormals, sourceNormalsSize};
    serializedData.targetPoints = {targetPoints, targetPointsSize};
    serializedData.targetNormals = {targetNormals, targetNormalsSize};
    serializedData.corres = {corres, corresSize};
}

void PointCloudRegistration::convertToGeom(
    const open3d::geometry::PointCloud& source,
    const open3d::geometry::PointCloud& target)
{
    // Add points, colors, and normals to the PointCloud
    auto sourcePcd = new Pcd{source.points_, source.normals_};
    auto targetPcd = new Pcd{target.points_, target.normals_};

    // Create a KDTree to efficiently find nearest neighbors
    open3d::geometry::KDTreeFlann kdtree_target(target);

    // Find correspondences between points in the source and target point clouds
    std::vector<Eigen::Vector2i> corrSet;
    std::vector<int> indices;
    std::vector<double> distances;
    for (int i = 0; i < source.points_.size(); ++i) {
        auto query_point = source.points_[i];
        kdtree_target.SearchKNN(query_point, 1, indices, distances);
        corrSet.push_back(Eigen::Vector2i(i, indices[0]));
    }
    geometry = Geometry{sourcePcd, targetPcd, &corrSet};
}

int PointCloudRegistration::performICPRegistration()
{
    // Initialize board communication
    unsigned int boardCount;

    std::thread test_thread[num_threads];
    std::thread _test_thread;
    // int* _test_ret;
    std::future<int> _test_future[num_threads];
    int cluster_id, nm_id;
    int i, ret;

    ret = pipe(pipe_fd);
    if (ret < 0)
        return 1;

    #ifdef USE_CHAN_EDCL
        if (PL_SetChannelEDCL(HOST_MAC_ADDRESS, BOARD_MAC_ADDRESS) != PL_OK) {
            printf("ERROR: Failed set channel EDCL!\n");
            return 2;
        }

        boardCount = 1;
    #else
        if (PL_GetBoardCount(&boardCount) != PL_OK) {
            printf("ERROR: Failed get count of boards!\n");
            return 2;
        }
    #endif

    if (boardCount < 1) {
        printf("ERROR: Failed find board!\n");
        return 3;
    }

    if (PL_GetBoardDesc(0, &board) != PL_OK) {
        printf("ERROR: Failed open board!\n");
        return 4;
    }

    #if 1
        if (PL_ResetBoard(board) != PL_OK) {
            printf("ERROR: Failed reset board!\n");
            PL_CloseBoardDesc(board);
            return 5;
        }

        if (PL_LoadInitCode(board) != PL_OK) {
            printf("ERROR: Failed load init code!\n");
            PL_CloseBoardDesc(board);
            return 6;
        }
    #endif

    // const int num_threads = 2;
    int iteration_num = geometry.corres->size();
    int iterations_per_thread = iteration_num / num_threads;
    
    for (i = 0; i < num_threads; i++) {
        if (i == 0) {
            cluster_id = -1;
            nm_id = -1;
        } else if (i == 1) {
            cluster_id = 0;
            nm_id = -1;
        } else if (i == 6) {
            cluster_id = 1;
            nm_id = -1;
        } else if (i == 11) {
            cluster_id = 2;
            nm_id = -1;
        } else if (i == 16) {
            cluster_id = 3;
            nm_id = -1;
        }

        all_core[i].cluster_id = cluster_id;
        all_core[i].nm_id = nm_id;

        test_promise[i] = std::promise<int>();
        _test_future[i] = test_promise[i].get_future();

        int start = i * iterations_per_thread;
        int end = (i == num_threads - 1) ? iteration_num : (i + 1) * iterations_per_thread;

        test_thread[i] = std::thread(calc_pcd, i, start, end,
            serializedData,
            std::ref(test_promise[i]));
        nm_id++;
    }

    sleep(1);

    pthread_mutex_lock(&work_mutex);
    do_wait = 0;
    pthread_cond_broadcast(&wait_cond);
    pthread_mutex_unlock(&work_mutex);
    i = 0;
    std::cout << "before while " << std::endl;

    std::vector<MatrixData> results;

    while (1) {
        ret = read(pipe_fd[0], &_test_thread, sizeof(std::thread));
        if (ret != sizeof(std::thread))
            exit(8);

        _test_thread.join();
        // get status
        if (_test_future[i].get() == 0)
        {
            // iterate successful threade num
            i++;
            auto result = test_promises[i].get_future().get();
            results.push_back(result);

        } else {
            std::cerr << "Error occurred in thread " << i << std::endl;
        }
        pthread_mutex_lock(&work_mutex);
        in_progress--;

        if (!in_progress) {
            pthread_mutex_unlock(&work_mutex);
            break;
        }
        pthread_mutex_unlock(&work_mutex);
    }

    PL_CloseBoardDesc(board);

    if (i != num_threads)
        return -1;

    // integrate the result
    std::tuple<MatrixData> integratedSum;
    for(i = 0; i < num_threads; ++i) {
        integratedSum.matrix.noalias() += results[i].matrix.noalias();
        integratedSum.vector.noalias() += results[i].vector.noalias();
        integratedSum.r2_sum += results[i].r2_sum;
    }

    result = integratedSum;
    printf("OK: test successful!\n");
    return 0;
}

// Function to fill the tuple
MatrixData fillMatrixData(int matrixSize, int vectorSize, PL_Word* matrix, PL_Word* vector, double r2_sum) {
    MatrixData data;

    // Fill the Eigen matrices from input arrays
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            // Assuming PL_Word is convertible to double
            data.matrix(i, j) = static_cast<double>(matrix[i * 6 + j]);
        }
    }

    for (int i = 0; i < 6; ++i) {
        // Assuming PL_Word is convertible to double
        data.vector(i) = static_cast<double>(vector[i]);
    }

    // Set the r2_sum value
    data.r2_sum = r2_sum;

    return data;
}

MatrixData PointCloudRegistration::getResult() const
{
    return result;
}

MatrixData calc_pcd(
	int index, int start, int end,
	const SerializedData& data,
	std::promise<int>& _test_promise)
{
    PL_Access* access = nullptr;
	pthread_t	self_thread;
	int		k, l;
	int		ret;
	bool done = false;

	pthread_mutex_lock(&work_mutex);
	in_progress++;
	pthread_mutex_unlock(&work_mutex);

    pthread_mutex_lock(&print_mutex);
    std::cout << "Thread[" << index << "] - Load program!" << std::endl;
	pthread_mutex_unlock(&print_mutex);
	// std::cout << sourceNormals[sourceNormalsSize - 1] << std::endl;

    if (PL_GetAccess(board, &all_core[index], &access) != PL_OK) {
		pthread_mutex_lock(&print_mutex);
		std::cerr << "ERROR[" << index << "]: Failed get access!" << std::endl;
		pthread_mutex_unlock(&print_mutex);
		goto err_sync_exit;
	}

	if (all_core[index].nm_id < 0)
		if (all_core[index].cluster_id == -1)
			ret = PL_LoadProgramFile(access, CCPU_PART_FILE_NAME);
		else
			ret = PL_LoadProgramFile(access, PC_PART_FILE_NAME);
	else
		ret = PL_LoadProgramFile(access, NM_PART_FILE_NAME);

	if (ret != PL_OK) {
		pthread_mutex_lock(&print_mutex);
		printf("ERROR[%d]: Failed load program\n", index);
		pthread_mutex_unlock(&print_mutex);

		goto err_sync_exit;
	}

	pthread_mutex_lock(&work_mutex);
	if (do_wait)
		pthread_cond_wait(&wait_cond, &work_mutex);
	pthread_mutex_unlock(&work_mutex);

	pthread_mutex_lock(&print_mutex);
	printf("Thread[%d] - Start program with start = %d, end = %d!\n", index, start, end);
	pthread_mutex_unlock(&print_mutex);
	
	{
		DWORD Count = 0;
		PL_Addr AddrLoad = 0;
		PL_Addr AddrStoreSourcePoints = 0;
		PL_Addr AddrStoreSourceNormals = 0;
		PL_Addr AddrStoreTargetPoints = 0;
		PL_Addr AddrStoreTargetNormals = 0;
		PL_Addr AddrStoreCorres = 0;
		PL_Word syncValue = 0;
		
		// receive data from here
		int sendStatus = 1;
		if (PL_SyncArray(access, sendStatus, 0, 0, &syncValue, &AddrLoad, &Count) != PL_OK) {
			pthread_mutex_lock(&print_mutex);
			printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
			pthread_mutex_unlock(&print_mutex);

			goto err_sync_exit;
		}

		pthread_mutex_lock(&print_mutex);
		printf("Thread[%d] - Received AddrLoad! Count = %d, sendStatus = %d, Sync = %d\n", index, Count, sendStatus, syncValue);
		pthread_mutex_unlock(&print_mutex);

		if (syncValue != 1) {
			pthread_mutex_lock(&print_mutex);
			printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
			pthread_mutex_unlock(&print_mutex);

			goto err_sync_exit;
		}

		pthread_mutex_lock(&print_mutex);
		if (index == 0) {
            printf("count = %d\n", Count);
		}
        pthread_mutex_unlock(&print_mutex);
			
		sendStatus = 2;

		if (PL_SyncArray(access, sendStatus, 0, data.sourcePoints.size, &syncValue, &AddrStoreSourcePoints, NULL) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);
				goto err_sync_exit;
        }

		pthread_mutex_lock(&print_mutex);
		printf("Thread[%d] - Received AddrStoreSourcePoints! sendStatus = %d, Sync = %d\n", index, sendStatus, syncValue);
		pthread_mutex_unlock(&print_mutex);
		
		// send data
		if (PL_WriteMemBlock(access, data.sourcePoints.buff, AddrStoreSourcePoints, data.sourcePoints.size) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);
				goto err_sync_exit;
        }

		sendStatus = 3;
		
		if (PL_Sync(access, sendStatus, &syncValue) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				goto err_sync_exit;
        }

		pthread_mutex_lock(&print_mutex);
		printf("Thread[%d] - Wrote source points! sendStatus = %d, Sync = %d\n", index, sendStatus, syncValue);
		pthread_mutex_unlock(&print_mutex);
		sendStatus = 4;

		if (PL_SyncArray(access, sendStatus, 0, data.sourceNormals.size, &syncValue, &AddrStoreSourceNormals, NULL) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);
				goto err_sync_exit;
        }

		pthread_mutex_lock(&print_mutex);
		printf("Thread[%d] - Received AddrStoreSource! sendStatus = %d, Sync = %d\n", index, sendStatus, syncValue);
		pthread_mutex_unlock(&print_mutex);
		
		// send data
		if (PL_WriteMemBlock(access, data.sourceNormals.buff, AddrStoreSourceNormals, data.sourceNormals.size) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);
				goto err_sync_exit;
        }
		done = true;
		
		sendStatus = 5;
		
		if (PL_Sync(access, sendStatus, &syncValue) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				goto err_sync_exit;
        }

		pthread_mutex_lock(&print_mutex);
		printf("Thread[%d] - Wrote source normals! sendStatus = %d, Sync = %d\n", index, sendStatus, syncValue);
		pthread_mutex_unlock(&print_mutex);

		if (syncValue != sendStatus) {
		        pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				goto err_sync_exit;
        }

		// send target points
		
		sendStatus = 6;
		if (PL_SyncArray(access, sendStatus, 0, data.targetPoints.size, &syncValue, &AddrStoreTargetPoints, NULL) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				goto err_sync_exit;
		}

		pthread_mutex_lock(&print_mutex);
		printf("Thread[%d] - Received AddrStoreTarget! Sync = %d\n", index, syncValue);
		pthread_mutex_unlock(&print_mutex);

		// send data
		if (PL_WriteMemBlock(access, data.targetPoints.buff, AddrStoreTargetPoints, data.targetPoints.size) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				goto err_sync_exit;
		}

		sendStatus = 7;
		int ret = PL_Sync(access, sendStatus, &syncValue);
		if (ret != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				
				goto err_sync_exit;
		}

		if (syncValue != sendStatus) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d! syncValue: %d, ret: %d, PL_OK: %d\n", index, sendStatus, syncValue, ret, PL_OK);
				pthread_mutex_unlock(&print_mutex);

				
				goto err_sync_exit;
		}

		pthread_mutex_lock(&print_mutex);
		printf("Thread[%d] - Wrote target pcd! sendStatus = %d, Sync = %d\n", index, sendStatus, syncValue);
		pthread_mutex_unlock(&print_mutex);

		// send target normals

		sendStatus = 8;
		if (PL_SyncArray(access, sendStatus, 0, data.targetNormals.size, &syncValue, &AddrStoreTargetNormals, NULL) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				goto err_sync_exit;
		}

		pthread_mutex_lock(&print_mutex);
		printf("Thread[%d] - Received AddrStoreTarget! Sync = %d\n", index, syncValue);
		pthread_mutex_unlock(&print_mutex);

		// send data
		if (PL_WriteMemBlock(access, data.targetNormals.buff, AddrStoreTargetNormals, data.targetNormals.size) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				goto err_sync_exit;
		}

		sendStatus = 9;
		ret = PL_Sync(access, sendStatus, &syncValue);
		if (ret != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				
				goto err_sync_exit;
		}

		if (syncValue != sendStatus) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d! syncValue: %d, ret: %d, PL_OK: %d\n", index, sendStatus, syncValue, ret, PL_OK);
				pthread_mutex_unlock(&print_mutex);

				
				goto err_sync_exit;
		}

		pthread_mutex_lock(&print_mutex);
		printf("Thread[%d] - Wrote target normals! sendStatus = %d, Sync = %d\n", index, sendStatus, syncValue);
		pthread_mutex_unlock(&print_mutex);

		// send corres

		sendStatus = 10;
		if (PL_SyncArray(access, sendStatus, 0, data.corres.size, &syncValue, &AddrStoreCorres, NULL) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				goto err_sync_exit;
		}

		pthread_mutex_lock(&print_mutex);
		printf("Thread[%d] - Received AddrStoreCorres! Sync = %d\n", index, syncValue);
		pthread_mutex_unlock(&print_mutex);

		// send data
		if (PL_WriteMemBlock(access, data.corres.buff, AddrStoreCorres, data.corres.size) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				goto err_sync_exit;
		}

		sendStatus = 11;
		ret = PL_Sync(access, sendStatus, &syncValue);
		if (ret != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				
				goto err_sync_exit;
		}

		if (syncValue != sendStatus) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d! syncValue: %d, ret: %d, PL_OK: %d\n", index, sendStatus, syncValue, ret, PL_OK);
				pthread_mutex_unlock(&print_mutex);

				
				goto err_sync_exit;
		}

		pthread_mutex_lock(&print_mutex);
		printf("Thread[%d] - Wrote target normals! sendStatus = %d, Sync = %d\n", index, sendStatus, syncValue);
		pthread_mutex_unlock(&print_mutex);
		
		// sending start and end
		sendStatus = start;
		if (PL_Sync(access, sendStatus, &syncValue) != PL_OK) {
		 		pthread_mutex_lock(&print_mutex);
		 		printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
		 		pthread_mutex_unlock(&print_mutex);

				
		 		goto err_sync_exit;
         }

		sendStatus = end;
		if (PL_Sync(access, sendStatus, &syncValue) != PL_OK) {
		 		pthread_mutex_lock(&print_mutex);
		 		printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
		 		pthread_mutex_unlock(&print_mutex);

				
		 		goto err_sync_exit;
		}


		// Receive results
		DWORD matrixSize = 0;
		PL_Addr AddrLoadMatrix = 0;

		// receive matrix
		sendStatus = 12;
		if (PL_SyncArray(access, sendStatus, 0, 0, &syncValue, &AddrLoadMatrix, &matrixSize) != PL_OK) {
			pthread_mutex_lock(&print_mutex);
			printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
			pthread_mutex_unlock(&print_mutex);

			goto err_sync_exit;
		}

		pthread_mutex_lock(&print_mutex);
		printf("Thread[%d] - Received AddrLoadMatirx! size = %d, sendStatus = %d, Sync = %d\n", index, matrixSize, sendStatus, syncValue);
		pthread_mutex_unlock(&print_mutex);

		if (syncValue != 1) {
			pthread_mutex_lock(&print_mutex);
			printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
			pthread_mutex_unlock(&print_mutex);

			goto err_sync_exit;
		}

		auto matrix = new PL_Word[matrixSize];

		if (PL_ReadMemBlock(access, matrix, AddrLoadMatrix, matrixSize) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				delete[] matrix;
				goto err_sync_exit;
		}
		sendStatus = 13;
		if (PL_Sync(access, sendStatus, &syncValue) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				
				goto err_sync_exit;
		}
        

		// receive vector

		DWORD vectorSize = 0;
		PL_Addr AddrLoadVector = 0;

		sendStatus = 14;
		if (PL_SyncArray(access, sendStatus, 0, 0, &syncValue, &AddrLoadVector, &vectorSize) != PL_OK) {
			pthread_mutex_lock(&print_mutex);
			printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
			pthread_mutex_unlock(&print_mutex);

			goto err_sync_exit;
		}

		pthread_mutex_lock(&print_mutex);
		printf("Thread[%d] - Received AddrLoadVector! size = %d, sendStatus = %d, Sync = %d\n", index, vectorSize, sendStatus, syncValue);
		pthread_mutex_unlock(&print_mutex);

		if (syncValue != 1) {
			pthread_mutex_lock(&print_mutex);
			printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
			pthread_mutex_unlock(&print_mutex);

			goto err_sync_exit;
		}

		auto vector = new PL_Word[vectorSize];

		if (PL_ReadMemBlock(access, vector, AddrLoadVector, vectorSize) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				delete[] vector;
				goto err_sync_exit;
		}
		sendStatus = 15;
		if (PL_Sync(access, sendStatus, &syncValue) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				
				goto err_sync_exit;
		}


		// receive r2_sum

		PL_Word r2_sum = 0;
		sendStatus = 16;
		if (PL_Sync(access, sendStatus, &r2_sum) != PL_OK) {
				pthread_mutex_lock(&print_mutex);
				printf("ERROR[%d]: ERROR in sync %d!\n", index, sendStatus);
				pthread_mutex_unlock(&print_mutex);

				
				goto err_sync_exit;
		}

		pthread_mutex_lock(&print_mutex);
		printf("Test[%d] - Received r2_sum = %d!\n", index, r2_sum);
		pthread_mutex_unlock(&print_mutex);

        auto result = fillMatrixData(matrixSize, vectorSize, matrix, vector, r2_sum);

		done = true;
		pthread_mutex_lock(&print_mutex);
		printf("Test[%d] - done!\n", index);
		pthread_mutex_unlock(&print_mutex);
	}

	// test_ret[index] = 0;
	// _test_promise.set_value(done ? 0 : -1);
    _test_promise.set_value(result);

    
err_sync_exit:
	if (access)
		PL_CloseAccess(access);

	self_thread = pthread_self();

	ret = write(pipe_fd[1], &self_thread, sizeof(pthread_t));
	if (ret != sizeof(pthread_t)) {
		std::cout << "write failed!\n";
		exit (8);
	}
    return done;

	// sleep(1);
	// _test_promise.set_value(-1);
}
