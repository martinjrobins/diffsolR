use std::sync::{Arc, Mutex, MutexGuard};
use diffsol::{matrix::{DenseMatrix, MatrixCommon}, DiffSl, OdeBuilder, OdeSolverMethod, OdeSolverProblem, OdeSolverState, StateRef};

type T = <M as MatrixCommon>::T;
type V = <M as MatrixCommon>::V;
type DM = <V as DefaultDenseMatrix>::M;
type CG = CraneliftModule;
type Eqn = DiffSl<M, CG>;
type Nls = NewtonNonlinearSolver<M, LS>;


pub struct OdeBuilderR(OdeBuilder<M>);

#[extendr]
impl OdeBuilderR {
    pub fn new() -> OdeBuilderR {
        OdeBuilderR(OdeBuilder::new())
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

    pub fn build(&mut self, code: &str) -> Result<OdeSolverProblemR> {
        let diffsl = DiffSl::compile(code).map_err(DiffsolErrorR)?;
        let mut problem = Err(DiffsolErrorR(DiffsolError::Other("Invalid builder".to_string())));
        take_mut::take(&mut self.0, |s| {
            problem = s.build_from_eqn(diffsl).map_err(DiffsolErrorR);
            OdeBuilder::new()
        });
        let problem = problem?;
        Ok(OdeSolverProblemR(Arc::new(Mutex::new(problem))))
    }
}

type ProblemHandle = Arc<Mutex<OdeSolverProblem<Eqn>>>;
type BdfState = <BdfSolver<'static> as OdeSolverMethod<'static, Eqn>>::State;


pub struct OdeSolverProblemR(ProblemHandle);

#[extendr]
impl OdeSolverProblemR {
    pub fn bdf(&self) -> Result<BdfSolverR> {
        let problem = Arc::clone(&self.0);
        let child: BdfSolver<'static> = {
            let problem_unlocked = problem.lock().unwrap();
            let child = problem_unlocked.bdf::<LS>().map_err(DiffsolErrorR)?;
            unsafe {
                std::mem::transmute(child)
            }
        };
        let child = Arc::new(Mutex::new(child));
        Ok(BdfSolverR(Dependent { _problem: problem, child }))
    }
    
}

pub struct BdfSolverR(Dependent<BdfSolver<'static>, OdeSolverProblem<Eqn>>);

#[extendr]
impl BdfSolverR {
    pub fn solve(&mut self, t1: T) -> Result<SolveSolutionR> {
        let mut s = self.0.lock()?;
        let (out, times) = s.solve(t1).map_err(DiffsolErrorR)?;
        Ok(SolveSolutionR{out, times})
    }
    
    pub fn solve_dense(&mut self, times: &[T]) -> Result<DenseSolveSolutionR> {
        let mut s = self.0.lock()?;
        let out = s.solve_dense(times).map_err(DiffsolErrorR)?;
        Ok(DenseSolveSolutionR{out})
    }

    pub fn state(&self) -> StateR {
        StateR(State::Bdf(self.0.clone()))
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


pub struct DenseMatrixR(DM);

#[extendr]
impl DenseMatrixR {
    pub fn new(rows: usize, cols: usize) -> DenseMatrixR {
        let m = DM::zeros(rows, cols);
        DenseMatrixR(m)
    }
    pub fn col(&self, i: usize) -> Doubles {
        self.0.column(i).iter().map(Into::into).collect::<Doubles>()
    }
}

#[extendr]
pub struct VectorR(V);

impl VectorR {
    pub fn new(size: usize) -> VectorR {
        let v = V::zeros(size);
        VectorR(v)
    }
    pub fn as_array(&self) -> Doubles {
        self.0.iter().map(Into::into).collect::<Doubles>()
    }
}

pub struct SolveSolutionR {
    out: DM,
    times: Vec<T>,
}

#[extendr]
impl SolveSolutionR {
    pub fn out(&self, i: usize) -> Doubles {
        self.out.column(i).iter().map(Into::into).collect::<Doubles>()
    }
    pub fn times(&self) -> Doubles {
        self.times.iter().map(Into::into).collect::<Doubles>()
    }
}

pub struct DenseSolveSolutionR {
    out: DM,
}

#[extendr]
impl DenseSolveSolutionR {
    pub fn out(&self, i: usize) -> Doubles {
        self.out.column(i).iter().map(Into::into).collect::<Doubles>()
    }
}




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


pub struct StateR(State);

impl StateR {
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
impl StateR {
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

