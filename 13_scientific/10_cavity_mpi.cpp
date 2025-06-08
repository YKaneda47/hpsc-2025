#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>

using namespace std::chrono;
using namespace std;

typedef vector<vector<float>> matrix;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2. / (nx - 1);
    double dy = 2. / (ny - 1);
    double dt = .01;
    double rho = 1.;
    double nu = .02;

    int local_nx_base = nx / size;
    int remainder = nx % size;

    int begin_idx;
    int end_idx;

    if (rank < remainder) {
        begin_idx = rank * (local_nx_base + 1);
        end_idx = begin_idx + (local_nx_base + 1);
    }
    else {
        begin_idx = rank * local_nx_base + remainder;
        end_idx = begin_idx + local_nx_base;
    }

    matrix u(ny, vector<float>(nx));
    matrix v(ny, vector<float>(nx));
    matrix p(ny, vector<float>(nx));
    matrix b(ny, vector<float>(nx));
    matrix un(ny, vector<float>(nx));
    matrix vn(ny, vector<float>(nx));
    matrix pn(ny, vector<float>(nx));

    for (int j = 0; j < ny; j++) {
        for (int i = begin_idx; i < end_idx; i++) {
            u[j][i] = 0;
            v[j][i] = 0;
            p[j][i] = 0;
            b[j][i] = 0;
        }
    }
    auto start = high_resolution_clock::now();

    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");

    vector<float> send_left_buffer(ny);
    vector<float> recv_left_buffer(ny);
    vector<float> send_right_buffer(ny);
    vector<float> recv_right_buffer(ny);

    for (int n = 0; n < nt; n++) {
        for (int j = 0; j < ny; j++) {
            send_right_buffer[j] = u[j][end_idx - 1];
            send_left_buffer[j] = u[j][begin_idx];
        }

        if (rank > 0) {
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 0,
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
                u[j][begin_idx - 1] = recv_left_buffer[j];
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 1,
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
                u[j][end_idx] = recv_right_buffer[j];
        }

        for (int j = 0; j < ny; j++) {
            send_right_buffer[j] = v[j][end_idx - 1];
            send_left_buffer[j] = v[j][begin_idx];
        }
        if (rank > 0) {
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 2,
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 3,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
                v[j][begin_idx - 1] = recv_left_buffer[j];
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 3,
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
                v[j][end_idx] = recv_right_buffer[j];
        }

        for (int j = 1; j < ny - 1; j++) {
            for (int i = max(1, begin_idx); i < min(nx - 1, end_idx); i++) {
                // Compute b[j][i]
                b[j][i] = rho * (1. / dt * ((u[j][i + 1] - u[j][i - 1]) / (2. * dx) + (v[j + 1][i] - v[j - 1][i]) / (2. * dy)) -
                                 pow((u[j][i + 1] - u[j][i - 1]) / (2. * dx), 2) -
                                 2. * ((u[j + 1][i] - u[j - 1][i]) / (2. * dy) * (v[j][i + 1] - v[j][i - 1]) / (2. * dx)) -
                                 pow((v[j + 1][i] - v[j - 1][i]) / (2. * dy), 2));
            }
        }
        for (int it = 0; it < nit; it++) {
            for (int j = 0; j < ny; j++) {
                for (int i = begin_idx; i < end_idx; i++)
                    pn[j][i] = p[j][i];
            }

            for (int j = 0; j < ny; j++) {
                send_right_buffer[j] = pn[j][end_idx - 1];
                send_left_buffer[j] = pn[j][begin_idx];
            }

            if (rank > 0) {
                MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 4,
                             &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 5,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < ny; ++j)
                    pn[j][begin_idx - 1] = recv_left_buffer[j];
            }
            if (rank < size - 1) {
                MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 5,
                             &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 4,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < ny; ++j)
                    pn[j][end_idx] = recv_right_buffer[j];
            }

            for (int j = 1; j < ny - 1; j++) {
                for (int i = max(1, begin_idx); i < min(nx - 1, end_idx); i++) {
                    // Compute p[j][i]
                    p[j][i] = (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) +
                               dx * dx * (pn[j + 1][i] + pn[j - 1][i]) -
                               b[j][i] * dx * dx * dy * dy) /
                              (2. * (dx * dx + dy * dy));
                }
            }
            if (rank == 0) { 
                for (int j = 0; j < ny; j++)
                    p[j][0] = p[j][1];
            }
            if (rank == size - 1) {
                for (int j = 0; j < ny; j++)
                    p[j][nx - 1] = p[j][nx - 2];
            }
            for (int i = begin_idx; i < end_idx; i++) {
                p[0][i] = p[1][i];
                p[ny - 1][i] = 0;
            }
        } // end of nit loop

        for (int j = 0; j < ny; j++) {
            for (int i = begin_idx; i < end_idx; i++) {
                un[j][i] = u[j][i];
                vn[j][i] = v[j][i];
            }
        }

        for (int j = 0; j < ny; j++) {
            send_right_buffer[j] = un[j][end_idx - 1];
            send_left_buffer[j] = un[j][begin_idx];
        }

        if (rank > 0) {
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 0,
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
                un[j][begin_idx - 1] = recv_left_buffer[j];
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 1,
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
                un[j][end_idx] = recv_right_buffer[j];
        }

        for (int j = 0; j < ny; j++) {
            send_right_buffer[j] = vn[j][end_idx - 1];
            send_left_buffer[j] = vn[j][begin_idx];
        }
        if (rank > 0) {
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 2,
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 3,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
                vn[j][begin_idx - 1] = recv_left_buffer[j];
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 3,
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
                vn[j][end_idx] = recv_right_buffer[j];
        }

        for (int j = 1; j < ny - 1; j++) {
            for (int i = max(1, begin_idx); i < min(nx - 1, end_idx); i++) {
                // Compute u[j][i] and v[j][i]
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) - vn[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) - dt / (2. * rho * dx) * (p[j][i + 1] - p[j][i - 1]) + nu * dt / (dx * dx) * (un[j][i + 1] - 2. * un[j][i] + un[j][i - 1]) + nu * dt / (dy * dy) * (un[j + 1][i] - 2. * un[j][i] + un[j - 1][i]);

                v[j][i] = vn[j][i] - un[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) - dt / (2. * rho * dy) * (p[j + 1][i] - p[j - 1][i]) // ここはdyに修正
                          + nu * dt / (dx * dx) * (vn[j][i + 1] - 2. * vn[j][i] + vn[j][i - 1]) + nu * dt / (dy * dy) * (vn[j + 1][i] - 2. * vn[j][i] + vn[j - 1][i]);
            }
        }
        if (rank == 0) {
            for (int j = 0; j < ny; j++) {
                u[j][0] = 0;
                v[j][0] = 0;
            }
        }
        if (rank == size - 1) {
            for (int j = 0; j < ny; j++) {
                u[j][nx - 1] = 0;
                v[j][nx - 1] = 0;
            }
        }
        for (int i = begin_idx; i < end_idx; i++) {
            u[0][i] = 0;
            u[ny - 1][i] = 1;
            v[0][i] = 0;
            v[ny - 1][i] = 0;
        }

        if (n % 10 == 0) {
          matrix u_global, v_global, p_global;
          if (rank == 0) {
              u_global.assign(ny, vector<float>(nx));
              v_global.assign(ny, vector<float>(nx));
              p_global.assign(ny, vector<float>(nx));
          }

          vector<int> recvcounts(size);
          vector<int> displs(size);
          int current_displ = 0;
          for (int i = 0; i < size; i++) {
              recvcounts[i] = (i < remainder) ? (local_nx_base + 1) : local_nx_base;
              displs[i] = current_displ;
              current_displ += recvcounts[i];
          }

          int local_nx = (rank < remainder) ? (local_nx_base + 1) : local_nx_base;

          for (int j = 0; j < ny; ++j) {
              MPI_Gatherv(
                  &u[j][begin_idx], local_nx, MPI_FLOAT,
                  (rank == 0) ? &u_global[j][0] : NULL,
                  &recvcounts[0], &displs[0], MPI_FLOAT,
                  0, MPI_COMM_WORLD);                    

              MPI_Gatherv(
                  &v[j][begin_idx], local_nx, MPI_FLOAT,
                  (rank == 0) ? &v_global[j][0] : NULL,
                  &recvcounts[0], &displs[0], MPI_FLOAT,
                  0, MPI_COMM_WORLD);

              MPI_Gatherv(
                  &p[j][begin_idx], local_nx, MPI_FLOAT,
                  (rank == 0) ? &p_global[j][0] : NULL,
                  &recvcounts[0], &displs[0], MPI_FLOAT,
                  0, MPI_COMM_WORLD);
          }
          
          if (rank == 0) {
              for (int j = 0; j < ny; j++) {
                  for (int i = 0; i < nx; i++)
                      ufile << u_global[j][i] << " ";
              }
              ufile << "\n";
              for (int j = 0; j < ny; j++) {
                  for (int i = 0; i < nx; i++)
                      vfile << v_global[j][i] << " ";
              }
              vfile << "\n";
              for (int j = 0; j < ny; j++) {
                  for (int i = 0; i < nx; i++)
                      pfile << p_global[j][i] << " ";
              }
              pfile << "\n";
          }
        }
    } // end of nt loop

    ufile.close();
    vfile.close();
    pfile.close();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    if (rank == 0) {
        printf("Elapsed time: %lld ms\n", duration.count());
    }

    MPI_Finalize();
    return 0;
}

