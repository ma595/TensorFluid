#include <stdio.h>
#include <iomanip>
#include <iostream> 
#include <fstream>
#include <algorithm>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

// implement halo swapping
// MPI
using Eigen::MatrixXd;

int v[9][2] = { {1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, 0}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1} };

double t[9] = { 1./36, 1./9, 1./36, 1./9, 4./9, 1./9, 1./36, 1./9, 1./36 };

int nx = 1680;
int ny = 720;

int cx = nx / 4;
int cy = ny / 2;
int r  = ny / 9;

int Lx = nx - 1;
int Ly = ny - 1;

double uLB = 0.04;
double Re = 120;

int frequency = 1001;
int maxiter = 1001;
double nu = uLB * r / Re; 
double omega = 1. / (3. * nu + 0.5);

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;

std::vector<Eigen::MatrixXd> vel(2, Eigen::MatrixXd(nx, ny));

bool circle(int x, int y){
  return pow(x - cx, 2) + pow(y - cy, 2) - r*r < 0;
}

// works 
void generate_mask(MatrixXb& mask){
  #pragma omp parallel for 
  for (int i = 0; i < nx; i++){
    for (int j = 0; j < ny; j++){
      mask(i, j) = circle(i, j);   
    }
  }
}

double inivel(double d, double x, double y){
  double vel = (1-d) * uLB * (1 + 1e-4*sin(y/Ly*2*M_PI));
  /* std::cout << y << std::endl; */
  return vel;
}

void equilibrium(std::vector<Eigen::MatrixXd>& eq, Eigen::MatrixXd& rho, std::vector<Eigen::MatrixXd>& u);
// CHECK: Can this be done with a lambda?
void initialise(Eigen::MatrixXd& rho, std::vector<Eigen::MatrixXd>& u){
  // set the initial velocity 
  vel[0].setZero();
  vel[1].setZero();
  u[0].setZero();
  u[1].setZero();
  #pragma omp parallel for 
  for (int i = 0; i < nx; ++i){
    for (int j = 0; j < ny; ++j){
      vel[0](i,j) = inivel(0, i, j);
      u[0](i,j) = inivel(0, i, j);
      rho(i,j) = 1.;
    }
  }
  /* Eigen::Tensor<double, 3> t_3d(9,nx,ny); */  
}

void streaming(std::vector<Eigen::MatrixXd>& fin, std::vector<Eigen::MatrixXd>& fout){
  #pragma omp parallel for 
  for (int i = 0; i < nx; ++i){
    for (int j = 0; j < ny; ++j){
      for (int c = 0; c < 9; ++c){
        int next_x = i + v[c][0];
        /* std::cout << next_x << " " << i << std::endl; */
        if(next_x < 0)
          next_x = nx-1;
        if(next_x >= nx)
          next_x = 0;

        int next_y = j + v[c][1];
        if(next_y < 0)
          next_y = ny - 1;
        if(next_y >= ny)
          next_y = 0;

        fin[c](next_x,next_y) = fout[c](i,j);
      }
    }
  }
}

void density(std::vector<Eigen::MatrixXd>& fin, Eigen::MatrixXd& rho){
  rho.setZero();
  #pragma omp parallel for 
  for (int i = 0; i < 9; i++){
    rho += fin[i];
  }
}

void velocity(std::vector<Eigen::MatrixXd>& fin, Eigen::MatrixXd& rho, std::vector<Eigen::MatrixXd>& u){
  u[0].setZero();
  u[1].setZero();
  #pragma omp parallel for 
  for (int i = 0; i < 9; ++i){
    u[0] += v[i][0] * fin[i];
    u[1] += v[i][1] * fin[i];
  }
  u[0] = rho.cwiseInverse().cwiseProduct(u[0]);
  u[1] = rho.cwiseInverse().cwiseProduct(u[1]);
}


// CHECK that the multiplication is the correct kind 
void equilibrium(std::vector<Eigen::MatrixXd>& eq, Eigen::MatrixXd& rho, std::vector<Eigen::MatrixXd>& u){
  Eigen::MatrixXd usqr = 3/2. * (u[0].cwiseProduct(u[0]) + u[1].cwiseProduct(u[1]));
  Eigen::MatrixXd ones = Eigen::MatrixXd::Constant(nx, ny, 1.); 
  #pragma omp parallel for 
  for (int i = 0; i < 9; ++i){
      MatrixXd vu = 3 * (v[i][0] * u[0] + v[i][1] * u[1]);
      eq[i] =  t[i] * rho.cwiseProduct(ones + vu + 0.5 * vu.cwiseProduct(vu) - usqr);
  }
}

// recalcualte eq in here
void apply_bcs(std::vector<Eigen::MatrixXd>& eq, std::vector<Eigen::MatrixXd>& fin,\
    std::vector<Eigen::MatrixXd>& u, Eigen::MatrixXd& rho){
  // set density on the right boundary  
  for (int c = 6; c < 9; c++){
    for (int j = 0; j < ny; j++){
      fin[c](nx-1, j) = fin[c](nx-2, j);
    }
  }

  // calculate rho and density
  density(fin, rho);
  velocity(fin, rho, u);

  // reset initial velocity (global)
  // CHECK 
  for (int j = 0; j < ny; ++j){
    u[0](0,j) = vel[0](0,j);
  }

  Eigen::VectorXd first = Eigen::VectorXd::Zero(ny);
  Eigen::VectorXd second = Eigen::VectorXd::Zero(ny);
  /* 0), second(ny, 0); */

  for (int c = 3; c < 6; ++c){
    for (int j = 0; j < ny; ++j){
      first(j) += fin[c](0,j);
      second(j) += fin[c+3](0,j);
    }
  }
  // adjust initial density   
  for (int j = 0; j < ny; ++j){
    rho(0,j) = 1./(1 - u[0](0, j)) * (first(j) + 2 * second(j));
  }

  equilibrium(eq, rho, u);

  for (int c = 0; c < 3; ++c){
    for (int j = 0; j < ny; ++j){
      fin[c](0,j) = eq[c](0,j) + fin[8-c](0,j) - eq[8-c](0,j);
    }
  }
}

void output_to_file(int time, int frequency, std::vector<Eigen::MatrixXd> u,
    Eigen::MatrixXd rho, MatrixXb mask){
  std::cout << "output to file" << std::endl;
  std::stringstream namestream;
  namestream << "cpp_out/vel." << std::setfill('0') << std::setw(3)
    << time / frequency << ".dat";
 
  std::string name;
  namestream >> name;

  std::ofstream outfile;
  outfile.open(name);
  outfile << "# X\tY\tvel_x\tvel_y\trho\tmask\n";
  for (int i = 0; i < nx; ++i){
    for (int j = 0; j < ny; ++j){
      outfile << i << '\t' << j << '\t' << u[0](i,j) << 
        '\t' << u[1](i,j) << '\t' << rho(i,j) << '\t' << mask(i,j) << '\n';
    }
    outfile << "\n";
  }
  outfile.close();
}

int main(){
  /* int * a = &v[0][0]; */
  /* MatrixXd m1(3,3); */
  /* printf("\n%d\n ", *(a+1)); */
  /* double * b = t; */

  /* int * c = v[0]; */
  /* for (int i = 0; i < 18; i++) */ 
  /*   printf("%d\n ", *(c+i)); */

  MatrixXb mask(nx, ny);
  generate_mask(mask);

  std::vector<Eigen::MatrixXd > fin(9, Eigen::MatrixXd::Zero(nx, ny));
  std::vector<Eigen::MatrixXd > fout(9, Eigen::MatrixXd::Zero(nx, ny));
  std::vector<Eigen::MatrixXd> eq(9, Eigen::MatrixXd::Zero(nx, ny));

  std::vector<Eigen::MatrixXd > u(2, Eigen::MatrixXd::Zero(nx, ny));
  Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(nx, ny); 

  initialise(rho, u);
  equilibrium(eq, rho, u);
  fin = eq;
  /* std::cout << fin[0] << std::endl; */  
  for (int time = 0; time < maxiter; ++time){
    std::cout << "time = " << time << std::endl;

    // boundary conditions 
    apply_bcs(eq, fin, u, rho);

    // collision
    #pragma omp for
    for (int c = 0; c < 9; ++c)
      fout[c] = fin[c] - omega*(fin[c] - eq[c]);

    // do bounceback only on solid nodes  
    #pragma omp parallel for 
    for (int c = 0; c < 9; ++c){
      for (int i = 0; i < nx; ++i){
        for (int j = 0; j < ny; ++j){
          if (mask(i,j)){
            /* std::cout << fout[c](i,j) << std::endl; */
            /* std::cout << fin[8-c](i,j) << " " << i << " " << j << " " << c << std::endl; */
            fout[c](i,j) = fin[8-c](i,j);
          } 
        }
      }
    }

    // update fin
    /* std::vector<Eigen::MatrixXd> before = fin; */
    streaming(fin, fout);
    /* if (fin[0].isApprox(before[0])) */
    /*   std::cout << "TRUE" << std::endl; */
    
    // now do plotting 
    /* density(fin, rho); */
    /* velocity(fin, rho, u); */
    if (time % frequency == 0)
      output_to_file(time, frequency, u, rho, mask);

    
  }
  /* printf("%d", m1.begin()); */

  /* std::cout << mask.size()  << mask.rows() << mask.cols() << std::endl; */
  /* std::cout << mask << std::endl; */

  return 0;
}
