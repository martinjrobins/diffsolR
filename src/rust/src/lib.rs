use extendr_api::prelude::*;
mod create_bindings;
use diffsol::{CraneliftModule, error::DiffsolError, DiffSl, Bdf, NalgebraLU, NewtonNonlinearSolver, OdeBuilder, Op, Sdirk, FaerLU, DefaultDenseMatrix};
use nalgebra::DMatrix;
use faer::Mat;


struct DiffsolErrorR(DiffsolError);


impl From<DiffsolErrorR> for Error {
    fn from(err: DiffsolErrorR) -> Error {
        Error::Other(err.0.to_string())
    }
}

create_binding!(nalgebra_dense_lu_f64, DMatrix<f64>, NalgebraLU<f64>, OdeBuilder1, OdeSolverProblem1, Sdirk1, OdeSolverState1);
create_binding!(faer_dense_lu_f64, Mat<f64>, FaerLU<f64>, OdeBuilder2, OdeSolverProblem2, Sdirk2, OdeSolverState2);


/// Return string `"Hello world!"` to R.
/// @export
#[extendr]
fn hello_world() -> &'static str {
    "Hello world!"
}


extendr_module! {
    mod diffsolR;
    fn hello_world;
    use nalgebra_dense_lu_f64;
    use faer_dense_lu_f64;
}
