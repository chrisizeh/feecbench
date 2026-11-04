// Optional MFEM 4.8 C++ bridge for platforms without working PyMFEM wheels.
// Build against an installed serial MFEM; do not commit compiled executables.
// Input: V F, then V xyz rows, then F vertex-index rows. Output: independent
// MatrixMarket M0/M1/D0/K plus signed DoF maps, all in MFEM's native ordering.
#include "mfem.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace mfem;

void write_matrix(const SparseMatrix &matrix, const std::string &path)
{
   std::ofstream out(path);
   if (!out) { throw std::runtime_error("cannot open matrix output"); }
   out << "%%MatrixMarket matrix coordinate real general\n"
       << matrix.Height() << " " << matrix.Width() << " " << matrix.NumNonZeroElems() << "\n"
       << std::setprecision(17);
   const int *rows = matrix.GetI(), *cols = matrix.GetJ();
   const double *data = matrix.GetData();
   for (int i = 0; i < matrix.Height(); ++i)
   {
      for (int k = rows[i]; k < rows[i+1]; ++k)
      { out << i+1 << " " << cols[k]+1 << " " << data[k] << "\n"; }
   }
}

int main(int argc, char **argv)
{
   if (argc != 3) { std::cerr << "usage: mfem_surface_assembly INPUT OUTPUT_PREFIX\n"; return 2; }
   try
   {
      std::ifstream input(argv[1]);
      const std::string prefix(argv[2]);
      int vertices, triangles;
      if (!(input >> vertices >> triangles) || vertices < 4 || triangles < 4)
      { throw std::runtime_error("invalid mesh input"); }
      Mesh mesh(2, vertices, triangles, 0, 3);
      for (int i = 0; i < vertices; ++i)
      {
         double point[3];
         if (!(input >> point[0] >> point[1] >> point[2])) { throw std::runtime_error("invalid vertex"); }
         mesh.AddVertex(point);
      }
      for (int i = 0; i < triangles; ++i)
      {
         int tri[3];
         if (!(input >> tri[0] >> tri[1] >> tri[2])) { throw std::runtime_error("invalid triangle"); }
         for (int j = 0; j < 3; ++j)
         { if (tri[j] < 0 || tri[j] >= vertices) { throw std::runtime_error("vertex index out of range"); } }
         mesh.AddTriangle(tri, 1);
      }
      mesh.FinalizeTriMesh(1, 0, false);
      H1_FECollection h1(1, 2);
      ND_FECollection nd(1, 2);
      FiniteElementSpace scalar(&mesh, &h1), velocity(&mesh, &nd);
      BilinearForm m0(&scalar), m1(&velocity), stiffness(&scalar);
      m0.AddDomainIntegrator(new MassIntegrator);
      m1.AddDomainIntegrator(new VectorFEMassIntegrator);
      stiffness.AddDomainIntegrator(new DiffusionIntegrator);
      m0.Assemble(); m0.Finalize();
      m1.Assemble(); m1.Finalize();
      stiffness.Assemble(); stiffness.Finalize();
      DiscreteLinearOperator gradient(&scalar, &velocity);
      gradient.AddDomainInterpolator(new GradientInterpolator);
      gradient.Assemble(); gradient.Finalize();
      write_matrix(m0.SpMat(), prefix + "_m0.mtx");
      write_matrix(m1.SpMat(), prefix + "_m1.mtx");
      write_matrix(stiffness.SpMat(), prefix + "_stiffness.mtx");
      write_matrix(gradient.SpMat(), prefix + "_d0.mtx");
      std::ofstream vmap(prefix + "_vertices.txt"), emap(prefix + "_edges.txt");
      Array<int> dofs, edge_vertices;
      for (int i = 0; i < vertices; ++i)
      {
         scalar.GetVertexDofs(i, dofs);
         if (dofs.Size() != 1) { throw std::runtime_error("expected one H1 vertex DoF"); }
         vmap << dofs[0] << "\n";
      }
      for (int i = 0; i < mesh.GetNEdges(); ++i)
      {
         mesh.GetEdgeVertices(i, edge_vertices);
         velocity.GetEdgeDofs(i, dofs);
         if (dofs.Size() != 1) { throw std::runtime_error("expected one ND edge DoF"); }
         emap << edge_vertices[0] << " " << edge_vertices[1] << " " << dofs[0] << "\n";
      }
      std::ofstream metadata(prefix + "_version.txt");
      metadata << MFEM_VERSION_STRING << "\n";
      return 0;
   }
   catch (const std::exception &error)
   { std::cerr << error.what() << "\n"; return 1; }
}
