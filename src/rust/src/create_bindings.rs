/// Macro to wrap creation of modules from bindings.rs
///
/// Each binding created requires a module name, a matrix and solver type to
/// expose the underlying diffsol classes to python.
///
/// This approach could be considered compiler abuse as the implementation sets
/// the three required values and then pulls in private/bindings.rs which in
/// turn gets them from super::<name>. This is not pretty but it's the neatest
/// approach to coerce templated types into a common suite of module classes for
/// PyO3 without duplicating a lot of boilerplate and having upcasting issues.
#[macro_export]
macro_rules! create_binding {
    (
        $module_name:ident,
        $matrix_type:ty,
        $solver_type:ty,
        $builder_type:ident,
        $problem_type:ident,
        $bdf_solver_type:ident,
        $state_type:ident
    ) => {
        #[path="."]
        pub mod $module_name {
            use super::*;

            // Module name, underlying type name and type
            const MODULE_NAME: &'static str = "$module_name";
            type M = $matrix_type;
            type LS = $solver_type;

use std::sync::{Arc, Mutex, MutexGuard};
use diffsol::{matrix::{DenseMatrix, MatrixCommon}, DiffSl, OdeBuilder, OdeSolverMethod, OdeSolverProblem, OdeSolverState, StateRef};

type T = <M as MatrixCommon>::T;
type V = <M as MatrixCommon>::V;
type DM = <V as DefaultDenseMatrix>::M;
type CG = CraneliftModule;
type Eqn = DiffSl<M, CG>;
type Nls = NewtonNonlinearSolver<M, LS>;


pub struct $builder_type(OdeBuilder<M>);

#[extendr]
impl $builder_type {
    pub fn new() -> $builder_type {
        $builder_type(OdeBuilder::new())
    }

    pub fn t0(&mut self, t0: T) {
        take_mut::take(&mut self.0, |s| {
            s.t0(t0)
        });
    }

    pub fn h0(&mut self, h0: T) {
        take_mut::take(&mut self.0, |s| {
            s.h0(h0)
        });
    }

    pub fn p(&mut self, p: Doubles) {
        let pv = p.iter().map(|x| x.inner()).collect::<Vec<_>>();
        take_mut::take(&mut self.0, |s| {
            s.p(pv)
        });
    }

    pub fn build(&mut self, code: &str) -> Result<$problem_type> {
        let diffsl = DiffSl::compile(code).map_err(DiffsolErrorR)?;
        let mut problem = Err(DiffsolErrorR(DiffsolError::Other("Invalid builder".to_string())));
        take_mut::take(&mut self.0, |s| {
            problem = s.build_from_eqn(diffsl).map_err(DiffsolErrorR);
            OdeBuilder::new()
        });
        let problem = problem?;
        Ok($problem_type(Arc::new(Mutex::new(problem))))
    }
}

type ProblemHandle = Arc<Mutex<OdeSolverProblem<Eqn>>>;
type BdfState = <BdfSolver<'static> as OdeSolverMethod<'static, Eqn>>::State;


pub struct $problem_type(ProblemHandle);

#[extendr]
impl $problem_type {
    pub fn bdf(&self) -> Result<$bdf_solver_type> {
        let problem = Arc::clone(&self.0);
        let child: BdfSolver<'static> = {
            let problem_unlocked = problem.lock().unwrap();
            let child = problem_unlocked.bdf::<LS>().map_err(DiffsolErrorR)?;
            unsafe {
                std::mem::transmute(child)
            }
        };
        let child = Arc::new(Mutex::new(child));
        Ok($bdf_solver_type(Dependent { _problem: problem, child }))
    }
    
}

pub struct $bdf_solver_type(Dependent<BdfSolver<'static>, OdeSolverProblem<Eqn>>);

#[extendr]
impl $bdf_solver_type {
    pub fn solve(&mut self, t1: T) -> Result<List> {
        let mut s = self.0.lock()?;
        let (out, times) = s.solve(t1).map_err(DiffsolErrorR)?;
        let times = times.iter().map(Into::into).collect::<Doubles>();
        let mut out_m = RMatrix::<T>::new(out.nrows(), out.ncols());
        for i in 0..out.nrows() {
            for j in 0..out.ncols() {
                out_m[[i, j]] = out[(i, j)].into();
            }
        }
        let out = out_m;
        Ok(list!(
            out = out,
            times = times,
        ))
    }
    
    pub fn solve_dense(&mut self, times: &[T]) -> Result<RMatrix<T>> {
        let mut s = self.0.lock()?;
        let out = s.solve_dense(times).map_err(DiffsolErrorR)?;
        let mut out_m = RMatrix::<T>::new(out.nrows(), out.ncols());
        for i in 0..out.nrows() {
            for j in 0..out.ncols() {
                out_m[[i, j]] = out[(i, j)].into();
            }
        }
        Ok(out_m)
    }

    pub fn state(&self) -> $state_type {
        $state_type(State::Bdf(self.0.clone()))
    }
}


//pub struct SdirkSolverR(Dependent<SdirkSolver<'static>, OdeSolverProblem<Eqn>>);


struct Dependent<C, P> {
    child: Arc<Mutex<C>>,
    _problem: Arc<Mutex<P>>,
}

impl<C, P> Dependent<C, P> {
    fn lock(&self) -> Result<MutexGuard<'_, C>> {
        self.child.lock().map_err(|e|  Error::Other(e.to_string()))   
    }
}

impl<C, P> Clone for Dependent<C, P> {
    fn clone(&self) -> Self {
        Dependent {
            child: Arc::clone(&self.child),
            _problem: Arc::clone(&self._problem),
        }
    }
}

type BdfSolver<'a> = Bdf<'a, Eqn, Nls, DM>;
type SdirkSolver<'a> = Sdirk<'a, Eqn, LS, DM>;


enum State {
    BdfOwned(BdfState),
    Bdf(Dependent<BdfSolver<'static>, OdeSolverProblem<Eqn>>),
    SdirkOwned(BdfState),
    Sdirk(Dependent<SdirkSolver<'static>, OdeSolverProblem<Eqn>>),
    Empty(()),
}

impl Default for State {
    fn default() -> Self {
        State::Empty(())
    }
}


pub struct $state_type(State);

impl $state_type {
    fn read_vector(&self, f: impl FnOnce(StateRef<'_, V>) -> &V) -> Result<Doubles> {
        let res = match &self.0 {
            State::BdfOwned(s) => f(s.as_ref()).as_slice().iter().map(Into::into).collect::<Doubles>(),
            State::Bdf(s) => {
                let s = s.lock().map_err(|e| Error::Other(e.to_string()))?;
                let s = f(s.state()).as_slice();
                s.iter().map(Into::into).collect::<Doubles>()
            },
            State::SdirkOwned(s) => f(s.as_ref()).as_slice().iter().map(Into::into).collect::<Doubles>(),
            State::Sdirk(s) => {
                let s = s.lock().map_err(|e| Error::Other(e.to_string()))?;
                let s = f(s.state()).as_slice();
                s.iter().map(Into::into).collect::<Doubles>()
            },
            _ => panic!("Invalid state"),
        };
        Ok(res)
    }
    fn read_scalar(&self, f: impl FnOnce(StateRef<'_, V>) -> T) -> Result<Rfloat> {
        let res = match &self.0 {
            State::BdfOwned(s) => f(s.as_ref()),
            State::Bdf(s) => {
                let s = s.lock().map_err(|e| Error::Other(e.to_string()))?;
                f(s.state())
            },
            State::SdirkOwned(s) => f(s.as_ref()),
            State::Sdirk(s) => {
                let s = s.lock().map_err(|e| Error::Other(e.to_string()))?;
                f(s.state())
            },
            _ => panic!("Invalid state"),
        };
        Ok(res.into())
    }
}


#[extendr]
impl $state_type{
    pub fn y(&self) -> Result<Doubles> {
        self.read_vector(|s| s.y)
    }

    pub fn dy(&self) -> Result<Doubles> {
        self.read_vector(|s| s.dy)
    }

    pub fn t(&self) -> Result<Rfloat> {
        self.read_scalar(|s| s.t)
    }

    pub fn h(&self) -> Result<Rfloat> {
        self.read_scalar(|s| s.h)
    }
}



            extendr_module! {
                mod $module_name;
                impl $builder_type;
                impl $problem_type;
                impl $bdf_solver_type;
                impl $state_type;
            }

        }
    };
}