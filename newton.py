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


#iterator for the right hand side
def new_rhs_iter(mesh, quadrule, f, un, alpha):

  weights = quadrule.weights
  qpoints = quadrule.points
  shapeF = shape2D_LFE(quadrule)

  for (a, b, c), tri, BK, detBK in zip(mesh.points_iter(), mesh.triangles, mesh.BK, mesh.detBK):

    # push forward of the local quadpoints (c.f. mass matrix with reaction term).
    x = qpoints @ BK.T + a[_]

    un_loc = alpha*(shapeF @ un[tri])**3

    # rhs function f evaluated in the push-forward points
    fx = -f(x) + un_loc

    yield (shapeF * (weights * fx)[:, _]).sum(0) * detBK


#iterator for the mass matrix (same as in fixed point but we multiply by 3)
def mass_with_nonlinear_term_3(mesh, quadrule, un):
  weights = quadrule.weights
  shapeF = shape2D_LFE(quadrule)

  for tri, detBK in zip(mesh.triangles, mesh.detBK):

    un_loc = 3*(shapeF @ un[tri])**2

    outer = (weights[:, _, _] * un_loc[:, _, _] * shapeF[..., _] * shapeF[:, _]).sum(0)
    yield outer * detBK

#main with while loop for the iterations
def main(mesh, alpha=0.1):

  quadrule = seven_point_gauss_6()
  un = np.zeros((len(mesh.points),))
  
  A_iter = stiffness_with_diffusivity_iter(mesh, quadrule)
  A = assemble_matrix_from_iterables(mesh, A_iter)

  bindices = mesh.boundary_indices
  data = np.zeros(bindices.shape)

  #numbering the iterations
  i=1

  while True:
    M_iter = mass_with_nonlinear_term_3(mesh, quadrule, un)
    M = assemble_matrix_from_iterables(mesh, M_iter)
    f = - A @ un - assemble_rhs_from_iterables(mesh, new_rhs_iter(mesh, quadrule, lambda x: np.array([100]), un, alpha))
    S = alpha*M + A
    unp = solve_with_dirichlet_data(S, f, bindices, data) + un
    diff = np.linalg.norm(un-unp, np.inf)
    print(i, diff)
    if diff < 1e-6:
      break
    un=unp
    i+=1

  mesh.tripcolor(unp)


if __name__ == '__main__':
  
  square = np.array([ [0, 0],
                       [1, 0],
                       [1, 1],
                       [0, 1] ])
  mesh = Triangulation.from_polygon(square, mesh_size=0.1)
  main(mesh,5)