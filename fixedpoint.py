from integrate import stiffness_with_diffusivity_iter, \
                     assemble_matrix_from_iterables, \
                     assemble_rhs_from_iterables, \
                     stiffness_with_diffusivity_iter, \
                     poisson_rhs_iter, \
                     shape2D_LFE
from quad import seven_point_gauss_6
from solve import solve_with_dirichlet_data
from util import np, _
from mesh import Triangulation

#iterator for the mass matrix
def mass_with_nonlinear_term(mesh, quadrule, un):
  weights = quadrule.weights
  shapeF = shape2D_LFE(quadrule)

  for tri, detBK in zip(mesh.triangles, mesh.detBK):

    un_loc = (shapeF @ un[tri])**2

    outer = (weights[:, _, _] * un_loc[:, _, _] * shapeF[..., _] * shapeF[:, _]).sum(0)
    yield outer * detBK

#main with while loop for the iterations
def main(mesh, alpha=0.1):

  quadrule = seven_point_gauss_6()
  un = np.zeros((len(mesh.points),))
  
  A_iter = stiffness_with_diffusivity_iter(mesh, quadrule)
  A = assemble_matrix_from_iterables(mesh, A_iter)

  f = assemble_rhs_from_iterables(mesh, poisson_rhs_iter(mesh, quadrule, lambda x: np.array([100])))
  bindices = mesh.boundary_indices
  data = np.zeros(bindices.shape)

  #numbering the iterations
  i=1

  while True:
    M_iter = mass_with_nonlinear_term(mesh, quadrule, un)
    M = assemble_matrix_from_iterables(mesh, M_iter)
    S = alpha*M + A
    unp=solve_with_dirichlet_data(S, f, bindices, data)
    diff=np.linalg.norm(un-unp, np.inf)
    print(i, diff)
    if diff < 1e-6:
      break
    un=unp
    i+=1

  mesh.tripcolor(unp)


if __name__ == '__main__':
  
  square = np.array([  [0, 0],
                       [1, 0],
                       [1, 1],
                       [0, 1] ])
  mesh = Triangulation.from_polygon(square, mesh_size=0.1)
  main(mesh, 0.1)