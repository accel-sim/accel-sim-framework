pipeline {
    agent {
        label "purdue-cluster"
        }

    options {
        disableConcurrentBuilds()
    }

    stages {
        stage('setup-env') {
            steps{
                sh 'rm -rf env-setup && git clone git@github.com:purdue-aalp/env-setup.git &&\
                    cd env-setup && git checkout cluster-ubuntu'
            }
        }
        stage('accel-sim-build'){
            steps{
                sh '''#!/bin/bash -xe
                source ./env-setup/11.2.1_env_setup.sh
                rm -rf ./gpu-simulator/gpgpu-sim
                source ./gpu-simulator/setup_environment.sh
                make -j -C gpu-simulator
                make clean -C gpu-simulator
                make -j -C gpu-simulator'''
            }
        }
        stage('short-test'){
            steps{
                parallel "sass": {
                sh '''#!/bin/bash -xe
                source ./env-setup/11.2.1_env_setup.sh
                source ./gpu-simulator/setup_environment.sh
                ./util/job_launching/run_simulations.py -B rodinia_2.0-ft,GPU_Microbenchmark -C QV100-SASS -T ~/../common/accel-sim/traces/volta-tesla-v100/latest/ -N sass-short-${BUILD_NUMBER}
                ./util/job_launching/run_simulations.py -B rodinia_2.0-ft,GPU_Microbenchmark -C RTX2060-SASS -T ~/../common/accel-sim/traces/turing-rtx2060/latest/ -N sass-short-${BUILD_NUMBER}
                ./util/job_launching/run_simulations.py -B rodinia_2.0-ft,GPU_Microbenchmark -C RTX3070-SASS -T ~/../common/accel-sim/traces/ampere-rtx3070/latest/ -N sass-short-${BUILD_NUMBER}
                ./util/job_launching/monitor_func_test.py -I -v -s stats-per-app-sass.csv -N sass-short-${BUILD_NUMBER}'''
               }, "ptx": {
                sh '''#!/bin/bash -xe
                source ./env-setup/11.2.1_env_setup.sh
                source ./gpu-simulator/setup_environment.sh

                rm -rf ./gpu-app-collection
                git clone git@github.com:accel-sim/gpu-app-collection.git
                source ./gpu-app-collection/src/setup_environment
                make rodinia_2.0-ft GPU_Microbenchmark -j -C ./gpu-app-collection/src
                ./gpu-app-collection/get_regression_data.sh

                ./util/job_launching/run_simulations.py -B rodinia_2.0-ft,GPU_Microbenchmark -C QV100-PTX,RTX2060-PTX,RTX3070-PTX -N short-ptx-${BUILD_NUMBER}
                ./util/job_launching/monitor_func_test.py -I -v -s stats-per-app-ptx.csv -N short-ptx-${BUILD_NUMBER}'''
               }
            }
        }
        stage('archive-stats') {
            steps{
                sh '''#!/bin/bash -xe
                source ./env-setup/11.2.1_env_setup.sh
                rm -rf ./statistics-archive
                git clone git@github.com:accel-sim/statistics-archive.git
                # either create a new branch or check it out if it already exists
                git -C ./statistics-archive checkout ${JOB_NAME} 2>/dev/null || git -C ./statistics-archive checkout -b ${JOB_NAME}
                ./util/job_launching/get_stats.py -k -K -R -B GPU_Microbenchmark -C QV100-SASS -A | tee v100-ubench-sass-${BUILD_NUMBER}.csv
                ./util/job_launching/get_stats.py -k -K -R -B GPU_Microbenchmark -C RTX2060-SASS -A | tee turing-ubench-sass-${BUILD_NUMBER}.csv
                ./util/job_launching/get_stats.py -k -K -R -B GPU_Microbenchmark -C RTX3070-SASS -A | tee ampere-ubench-sass-${BUILD_NUMBER}.csv
                mkdir -p statistics-archive/ubench/
                ./util/plotting/merge-stats.py -R -c ./statistics-archive/ubench/v100-ubench-sass.csv,v100-ubench-sass-${BUILD_NUMBER}.csv \
                    | tee v100-ubench-sass.csv && mv v100-ubench-sass.csv ./statistics-archive/ubench/
                ./util/plotting/merge-stats.py -R -c ./statistics-archive/ubench/turing-ubench-sass.csv,turing-ubench-sass-${BUILD_NUMBER}.csv \
                    | tee turing-ubench-sass.csv && mv turing-ubench-sass.csv ./statistics-archive/ubench/
                ./util/plotting/merge-stats.py -R -c ./statistics-archive/ubench/ampere-ubench-sass.csv,ampere-ubench-sass-${BUILD_NUMBER}.csv \
                    | tee ampere-ubench-sass.csv && mv ampere-ubench-sass.csv ./statistics-archive/ubench/
                git -C ./statistics-archive add --all
                git -C ./statistics-archive commit \
                    -m "Jenkins automated checkin ${JOB_NAME} Build:${BUILD_NUMBER}" || echo "No Changes."
                git -C ./statistics-archive push -u origin ${JOB_NAME}
                '''
            }
        }
        stage('correlate-ubench'){
            steps{
                sh '''#!/bin/bash -xe
                source ./env-setup/11.2.1_env_setup.sh
                ./util/hw_stats/get_hw_data.sh
                rm -rf ./util/plotting/correl-html/
                ./util/plotting/plot-correlation.py -c ./statistics-archive/ubench/v100-ubench-sass.csv -H ./hw_run/QUADRO-V100/device-0/10.2/ | tee v100-ubench-correl.txt
                ./util/plotting/plot-correlation.py -c ./statistics-archive/ubench/turing-ubench-sass.csv -H ./hw_run/TURING-RTX2060/10.2/ | tee turing-ubench-correl.txt
                ./util/plotting/plot-correlation.py -c ./statistics-archive/ubench/ampere-ubench-sass.csv -H ./hw_run/AMPERE-RTX3070/11.2/ | tee ampere-ubench-correl.txt
                ssh tgrogers@dynamo mkdir -p /home/dynamo/a/tgrogers/website/accel-sim/latest-correl/${JOB_NAME}/
                rsync --delete -r ./util/plotting/correl-html/ tgrogers@dynamo:~/website/accel-sim/latest-correl/${JOB_NAME}/
                echo "Correlation Report at: https://engineering.purdue.edu/tgrogers/accel-sim/latest-correl/${JOB_NAME}/"
                '''
            }
        }
    }
    post {
        success {
            emailext body: "See ${BUILD_URL}.\n\nCorrelation at https://engineering.purdue.edu/tgrogers/accel-sim/latest-correl/${JOB_NAME}/",
                recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                    [$class: 'RequesterRecipientProvider']],
                subject: "[AALP Jenkins] Build #${BUILD_NUMBER} - Success!",
                to: 'tgrogers@purdue.edu'
        }
        failure {
            emailext body: "See ${BUILD_URL}",
                recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                    [$class: 'RequesterRecipientProvider']],
                subject: "[AALP Jenkins] Build #${BUILD_NUMBER} - ${currentBuild.result}",
                to: 'tgrogers@purdue.edu'
        }
   }
}
