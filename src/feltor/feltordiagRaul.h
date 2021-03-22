#include <string>
#include <vector>
#include <functional>

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"

#include "feltor/feltor.h"
#include "feltor/parameters.h"

#include "feltor/init.h"

namespace feltor{

// This file constitutes the diagnostics module for feltor
// The way it works is that it allocates global lists of Records that describe what goes into the file
// You can register you own diagnostics in one of three diagnostics lists (static 3d, dynamic 3d and
// dynamic 2d) further down
// which will then be applied during a simulation

namespace routines{

struct RadialParticleFlux{
    RadialParticleFlux( double tau, double mu):
        m_tau(tau), m_mu(mu){
    }
    //jsNC
    DG_DEVICE double operator()( double ne, double ue,
        double d0S, double d1S, double d2S, //Psip
        double curv0,       double curv1,       double curv2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        double curvKappaS = curvKappa0*d0S+curvKappa1*d1S+curvKappa2*d2S;
        double curvS = curv0*d0S+curv1*d1S+curv2*d2S;
        double JPsi =
            + ne * m_mu*ue*ue*curvKappaS
            + ne * m_tau*curvS;
        return JPsi;
    }
    //jsNA
    DG_DEVICE double operator()( double ne, double ue, double A,
        double d0A, double d1A, double d2A,
        double d0S, double d1S, double d2S, //Psip
        double b_0,         double b_1,         double b_2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        double curvKappaS = curvKappa0*d0S+curvKappa1*d1S+curvKappa2*d2S;
        double SA = b_0*( d1S*d2A-d2S*d1A)+
                    b_1*( d2S*d0A-d0S*d2A)+
                    b_2*( d0S*d1A-d1S*d0A);
        double JPsi =
            ne*ue* (A*curvKappaS + SA );
        return JPsi;
    }
    private:
    double m_tau, m_mu;
};

struct RadialEnergyFlux{
    RadialEnergyFlux( double tau, double mu, double z):
        m_tau(tau), m_mu(mu), m_z(z){
    }

    DG_DEVICE double operator()( double ne, double ue, double P,
        double d0P, double d1P, double d2P, //Phi
        double d0S, double d1S, double d2S, //Psip
        double b_0,         double b_1,         double b_2,
        double curv0,  double curv1,  double curv2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        double curvKappaS = curvKappa0*d0S+curvKappa1*d1S+curvKappa2*d2S;
        double curvS = curv0*d0S+curv1*d1S+curv2*d2S;
        double PS = b_0 * ( d1P * d2S - d2P * d1S )+
                    b_1 * ( d2P * d0S - d0P * d2S )+
                    b_2 * ( d0P * d1S - d1P * d0S );
        double JN =
            + ne * PS
            + ne * m_mu*ue*ue*curvKappaS
            + ne * m_tau*curvS;
        double Je = m_z*(m_tau * log(ne) + 0.5*m_mu*ue*ue + P)*JN
            + m_z*m_mu*m_tau*ne*ue*ue*curvKappaS;
        return Je;
    }
    DG_DEVICE double operator()( double ne, double ue, double P, double A,
        double d0A, double d1A, double d2A,
        double d0S, double d1S, double d2S, //Psip
        double b_0,         double b_1,         double b_2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        double curvKappaS = curvKappa0*d0S+curvKappa1*d1S+curvKappa2*d2S;
        double SA = b_0 * ( d1S * d2A - d2S * d1A )+
                    b_1 * ( d2S * d0A - d0S * d2A )+
                    b_2 * ( d0S * d1A - d1S * d0A );
        double JN = m_z*ne*ue* (A*curvKappaS + SA );
        double Je = m_z*( m_tau * log(ne) + 0.5*m_mu*ue*ue + P )*JN
                    + m_z*m_tau*ne*ue* (A*curvKappaS + SA );
        return Je;
    }
    //energy dissipation
    DG_DEVICE double operator()( double ne, double ue, double P,
        double lapMperpN, double lapMperpU){
        return m_z*(m_tau*(1+log(ne))+P+0.5*m_mu*ue*ue)*lapMperpN
                + m_z*m_mu*ne*ue*lapMperpU;
    }
    //energy source
    DG_DEVICE double operator()( double ne, double ue, double P,
        double source){
        return m_z*(m_tau*(1+log(ne))+P+0.5*m_mu*ue*ue)*source;
    }
    private:
    double m_tau, m_mu, m_z;
};









template<class Container>

void dot( const std::array<Container, 3>& v,
          const std::array<Container, 3>& w,
          Container& result)
{
    dg::blas1::evaluate( result, dg::equals(), dg::PairSum(),
        v[0], w[0], v[1], w[1], v[2], w[2]);
}
struct Times{
    DG_DEVICE void operator()(
            double lambda,
        double d0P, double d1P, double d2P, //any three vectors
        double d0S, double d1S, double d2S,
        double& c_0, double& c_1, double& c_2)
    {
        c_0 = lambda*(d1P*d2S-d2P*d1S);
        c_1 = lambda*(d2P*d0S-d0P*d2S);
        c_2 = lambda*(d0P*d1S-d1P*d0S);
    }
};

template<class Container>
void times(
          const std::array<Container, 3>& a,
          const std::array<Container, 3>& b,
          std::array<Container, 3>& c)
{
    dg::blas1::subroutine( Times(), 1.,
        a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}
template<class Container>
void times(
          const Container& lambda,
          const std::array<Container, 3>& a,
          const std::array<Container, 3>& b,
          std::array<Container, 3>& c)
{
    dg::blas1::subroutine( Times(), lambda,
        a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}
struct Jacobian{
    DG_DEVICE double operator()(
        double d0P, double d1P, double d2P, //any three vectors
        double d0S, double d1S, double d2S,
        double b_0, double b_1, double b_2)
    {
        return      b_0*( d1P*d2S-d2P*d1S)+
                    b_1*( d2P*d0S-d0P*d2S)+
                    b_2*( d0P*d1S-d1P*d0S);
    }
};
template<class Container>
void jacobian(
          const std::array<Container, 3>& a,
          const std::array<Container, 3>& b,
          const std::array<Container, 3>& c,
          Container& result)
{
    dg::blas1::evaluate( result, dg::equals(), Jacobian(),
        a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}
}//namespace routines





//From here on, we use the typedefs to ease the notation

struct Variables{
    feltor::Explicit<Geometry, IDMatrix, DMatrix, DVec>& f;
    feltor::Parameters p;
    dg::geo::TokamakMagneticField mag;
    dg::geo::Nablas nabla;
    std::array<DVec, 3> gradPsip;
    std::array<DVec, 3> tmp;
    std::array<DVec, 3> tmp2;
    std::array<DVec, 3> tmp3;
    DVec hoo; //keep hoo there to avoid pullback
};




struct Record{
    std::string name;
    std::string long_name;
    bool integral; //indicates whether the function should be time-integrated
    std::function<void( DVec&, Variables&)> function;
};




struct Record_static{
    std::string name;
    std::string long_name;
    std::function<void( HVec&, Variables&, Geometry& grid)> function;
};





///%%%%%%%%%%%%%%%%%%%%%%%EXTEND LISTS WITH YOUR DIAGNOSTICS HERE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
///%%%%%%%%%%%%%%%%%%%%%%%EXTEND LISTS WITH YOUR DIAGNOSTICS HERE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
///%%%%%%%%%%%%%%%%%%%%%%%EXTEND LISTS WITH YOUR DIAGNOSTICS HERE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//Here is a list of static (time-independent) 3d variables that go into the output
//Except xc, yc, and zc these are redundant since we have geometry_diag.cu
//MW: maybe it's a test of sorts
std::vector<Record_static> diagnostics3d_static_list = {
    { "BR", "R-component of magnetic field in cylindrical coordinates",
        []( HVec& result, Variables& v, Geometry& grid){
            dg::geo::BFieldR fieldR(v.mag);
            result = dg::pullback( fieldR, grid);
        }
    },
    { "BZ", "Z-component of magnetic field in cylindrical coordinates",
        []( HVec& result, Variables& v, Geometry& grid){
            dg::geo::BFieldZ fieldZ(v.mag);
            result = dg::pullback( fieldZ, grid);
        }
    },
    { "BP", "Contravariant P-component of magnetic field in cylindrical coordinates",
        []( HVec& result, Variables& v, Geometry& grid){
            dg::geo::BFieldP fieldP(v.mag);
            result = dg::pullback( fieldP, grid);
        }
    },
    { "Psip", "Flux-function psi",
        []( HVec& result, Variables& v, Geometry& grid){
             result = dg::pullback( v.mag.psip(), grid);
        }
    },
    { "Nprof", "Density profile (that the source may force)",
        []( HVec& result, Variables& v, Geometry& grid ){
            result = dg::evaluate( dg::zero, grid);
            bool fixed_profile;
            HVec source = feltor::source_profiles.at(v.p.source_type)(
                fixed_profile, result, grid, v.p, v.mag);
        }
    },
    { "Source", "Source region",
        []( HVec& result, Variables& v, Geometry& grid ){
            bool fixed_profile;
            HVec profile;
            result = feltor::source_profiles.at(v.p.source_type)(
                fixed_profile, profile, grid, v.p, v.mag);
        }
    },
    { "xc", "x-coordinate in Cartesian coordinate system",
        []( HVec& result, Variables& v, Geometry& grid ){
            result = dg::evaluate( dg::cooRZP2X, grid);
        }
    },
    { "yc", "y-coordinate in Cartesian coordinate system",
        []( HVec& result, Variables& v, Geometry& grid ){
            result = dg::evaluate( dg::cooRZP2Y, grid);
        }
    },
    { "zc", "z-coordinate in Cartesian coordinate system",
        []( HVec& result, Variables& v, Geometry& grid ){
            result = dg::evaluate( dg::cooRZP2Z, grid);
        }
    },
};

std::array<std::tuple<std::string, std::string, HVec>, 3> generate_cyl2cart( Geometry& grid)
{
    HVec xc = dg::evaluate( dg::cooRZP2X, grid);
    HVec yc = dg::evaluate( dg::cooRZP2Y, grid);
    HVec zc = dg::evaluate( dg::cooRZP2Z, grid);
    std::array<std::tuple<std::string, std::string, HVec>, 3> list = {{
        { "xc", "x-coordinate in Cartesian coordinate system", xc },
        { "yc", "y-coordinate in Cartesian coordinate system", yc },
        { "zc", "z-coordinate in Cartesian coordinate system", zc }
    }};
    return list;
}

// Here are all 3d outputs we want to have
std::vector<Record> diagnostics3d_list = {
    {"electrons", "electron density", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(0), result);
        }
    },
    {"ions", "ion density", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(1), result);
        }
    },
    {"Ue", "parallel electron velocity", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(0), result);
        }
    },
    {"Ui", "parallel ion velocity", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(1), result);
        }
    },
    {"potential", "electric potential", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.potential(0), result);
        }
    },
    {"induction", "parallel magnetic induction", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.induction(), result);
        }
    }
};

//Here is a list of static (time-independent) 2d variables that go into the output
//MW: These are redundant since we have geometry_diag.cu -> remove ? if geometry_diag works as expected (I guess it can also be a test of sorts)
//MW: if they stay they should be documented in feltor.tex
//( we make 3d variables here but only the first 2d slice is output)
std::vector<Record_static> diagnostics2d_static_list = {
    { "Psip2d", "Flux-function psi",
        []( HVec& result, Variables& v, Geometry& grid ){
            result = dg::pullback( v.mag.psip(), grid);
        }
    },
    { "Ipol", "Poloidal current",
        []( HVec& result, Variables& v, Geometry& grid ){
            result = dg::pullback( v.mag.ipol(), grid);
        }
    },
    { "Bmodule", "Magnetic field strength",
        []( HVec& result, Variables& v, Geometry& grid ){
            result = dg::pullback( dg::geo::Bmodule(v.mag), grid);
        }
    },
    { "Divb", "The divergence of the magnetic unit vector",
        []( HVec& result, Variables& v, Geometry& grid ){
            result = v.f.divb();
        }
    },
    { "InvB", "Inverse of Bmodule",
        []( HVec& result, Variables& v, Geometry& grid ){
            result = v.f.binv();
        }
    },
    { "CurvatureKappaR", "R-component of the Kappa B curvature vector",
        []( HVec& result, Variables& v, Geometry& grid ){
            result = v.f.curvKappa()[0];
        }
    },
    { "CurvatureKappaZ", "Z-component of the Kappa B curvature vector",
        []( HVec& result, Variables& v, Geometry& grid){
            result = v.f.curvKappa()[1];
        }
    },
    { "CurvatureKappaP", "Contravariant Phi-component of the Kappa B curvature vector",
        []( HVec& result, Variables& v, Geometry& grid){
            result = v.f.curvKappa()[2];
        }
    },
    { "DivCurvatureKappa", "Divergence of the Kappa B curvature vector",
        []( HVec& result, Variables& v, Geometry& grid){
            result = v.f.divCurvKappa();
        }
    },
    { "CurvatureR", "R-component of the curvature vector",
        []( HVec& result, Variables& v, Geometry& grid){
            result = v.f.curv()[0];
        }
    },
    { "CurvatureZ", "Z-component of the full curvature vector",
        []( HVec& result, Variables& v, Geometry& grid){
            result = v.f.curv()[1];
        }
    },
    { "CurvatureP", "Contravariant Phi-component of the full curvature vector",
        []( HVec& result, Variables& v, Geometry& grid){
            result = v.f.curv()[2];
        }
    },
    { "bphi", "Contravariant Phi-component of the magnetic unit vector",
        []( HVec& result, Variables& v, Geometry& grid){
            result = v.f.bphi();
        }
    },
    {"NormGradPsip", "Norm of gradient of Psip",
        []( HVec& result, Variables& v, Geometry& grid){
            result = dg::pullback(
                dg::geo::SquareNorm( dg::geo::createGradPsip(v.mag), dg::geo::createGradPsip(v.mag)), grid);
        }
    }
};
// and here are all the 2d outputs we want to produce (currently ~ 100)
std::vector<Record> diagnostics2d_list = {
    {"electrons", "Electron density", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(0), result);
        }
    },
    {"ions", "Ion gyro-centre density", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(1), result);
        }
    },
    {"Ue", "Electron parallel velocity", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(0), result);
        }
    },
    {"Ui", "Ion parallel velocity", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(1), result);
        }
    },
    {"potential", "Electric potential", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.potential(0), result);
        }
    },
    {"psi", "Ion potential psi", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.potential(1), result);
        }
    },
    {"induction", "Magnetic potential", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.induction(), result);
        }
    },
    ///-----------------------RAUL VORTICITY ADDITIONS-------------------///

    {"elec_vorticity", "Electric vorticity (as time derivative)", false, //FINAL
        []( DVec& result, Variables& v) {
			 dg::blas1::copy(v.f.gradP(0)[0], v.tmp[0]);
             dg::blas1::copy(v.f.gradP(0)[1], v.tmp[1]);
             dg::blas1::pointwiseDot( v.f.density(1), v.tmp[0]); 
             dg::blas1::pointwiseDot( v.f.density(1), v.tmp[1]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1]);    
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1]);  
             dg::tensor::multiply2d(v.f.projection(), v.tmp[0], v.tmp[1], v.tmp[0], v.tmp[1]); //to transform the vector from covariant to contravariant    //CHANGE THE METRICS TO projection
             v.nabla.div(v.tmp[0], v.tmp[1], result); 
             dg::blas1::scal(result, -1);
        }
    },
    
    {"dielec_vorticity", "Dielectric vorticity (as time derivative)", false, //FINAL
        []( DVec& result, Variables& v) { 
			 integral=true;         
			 dg::blas1::copy(v.f.gradN(1)[0], v.tmp[0]);
             dg::blas1::copy(v.f.gradN(1)[1], v.tmp[1]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1]);    
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1]);  
             dg::tensor::multiply2d(v.f.projection(), v.tmp[0], v.tmp[1], v.tmp[0], v.tmp[1]); //to transform the vector from covariant to contravariant    
             v.nabla.div(v.tmp[0], v.tmp[1], result);
             dg::blas1::scal(result, -1);
             dg::blas1::scal(result, v.p.tau[1]);
        }
    },
    
     {"elec_ tensor term_tt", "electric tensor term (time integrated)", true, //FINAL
        []( DVec& result, Variables& v) {            
             nabla.b_cross_v(v.f.gradP(0)[0], v.f.gradP(0)[1], v.tmp[0], v.tmp[1]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0], v.tmp[0]); //tmp[]=u_E
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1], v.tmp[1]);
             
             dg::blas1::pointwiseDot(v.f.density(1), v.f.gradP(0)[0], v.tmp2[0]); //N nabla_phi
             dg::blas1::pointwiseDot(v.f.density(1), v.f.gradP(0)[1], v.tmp2[1]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp2[0], v.tmp2[0]); 
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp2[1], v.tmp2[1]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp2[0], v.tmp2[0]); 
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp2[1], v.tmp2[1]);//tmp2[]=N grad phi/B^2 
             v.nabla.div(v.tmp2[0], v.tmp2[1], v.tmp[3]);
             nabla.v_dot_nabla_f(v.tmp2[0], v.tmp2[1], v.tmp[0], v.tmp3[0]); 
             nabla.v_dot_nabla_f(v.tmp2[0], v.tmp2[1], v.tmp[1], v.tmp3[1]); 
             dg::blas1::pointwiseDot(v.tmp[3], v.tmp2[0], v.tmp2[0]);//div(N grad(phi)/B^2)*u_E
             dg::blas1::pointwiseDot(v.tmp[3], v.tmp2[1], v.tmp2[1]);          
             dg::blas1::axpby(1,v.tmp2[0], 1, v.tmp3[0]);
             dg::blas1::axpby(1,v.tmp2[1], 1, v.tmp3[1]);
             nabla.div(v.tmp3[0], v.tmp3[1], result);
        }
    },
    
         {"dielec_ tensor term_tt", "Dielectric tensor term (time integrated)", true, //FINAL
        []( DVec& result, Variables& v) {            
             nabla.b_cross_v(v.f.gradP(0)[0], v.f.gradP(0)[1], v.tmp[0], v.tmp[1]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0], v.tmp[0]); //u_E
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1], v.tmp[1]);
             
             dg::blas1::pointwiseDivide(v.f.binv(), v.f.gradN(1)[0], v.tmp2[0]);
             dg::blas1::pointwiseDivide(v.f.binv(), v.tmp2[0], v.tmp2[0]);
             dg::blas1::pointwiseDivide(v.f.binv(), v.f.gradN(1)[1], v.tmp2[1]);
             dg::blas1::pointwiseDivide(v.f.binv(), v.tmp2[1], v.tmp2[1]);
             
             v.nabla.div(v.tmp2[0],v.tmp2[1], v.tmp[2]);
             
             nabla.v_dot_nabla_f(v.tmp2[0], v.tmp2[1], v.tmp[0], v.tmp2[2]); 
             nabla.v_dot_nabla_f(v.tmp2[0], v.tmp2[1], v.tmp[1], v.tmp2[1]); 
             
             dg::blas1::pointwiseDot(v.tmp[2], v.tmp[0], v.tmp[0]);
             dg::blas1::pointwiseDot(v.tmp[2], v.tmp[1], v.tmp[1]);          
             dg::blas1::axpby(1,v.tmp[0], 1, v.tmp2[2]);
             dg::blas1::axpby(1,v.tmp[1], 1, v.tmp2[1]);
             nabla.div(v.tmp2[2], v.tmp2[1], result);            
        }
    },
    
    {"par_current_term_tt", "Parallel current term (time integrated)", true, //FINAL
        []( DVec& result, Variables& v) { 
             dg::blas1::pointwiseDot(v.f.density(1), v.f.velocity(1), v.tmp[0]);
             dg::blas1::pointwiseDot(-1., v.f.density(0), v.f.velocity(0), 1., v.tmp[0]);  
             dg::geo::ds_centered( v.tmp[0], v.tmp[1]);
             dg::pointwiseDot(v.tmp[0], dg::geo::GradLnB(geom), v.tmp[0]);
             dg::blas1::axpbypgz(1, v.tmp[0], -1, v.tmp[1], 1, result)         
        }
    },
    
    {"mag_term_tt", "Magnetization term (time integrated)", true, //FINAL
        []( DVec& result, Variables& v) {
             dg::blas1::copy(dg::evaluate(dg:ones, m_g), v.tmp[0]);
             dg::blas1::scal(v.tmp[0], 0.5);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0], v.tmp[0]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0], v.tmp[0]);//1/2B^2
             
             nabla.b_cross_v (v.f.gradA()[0], v.f.gradA()[1], v.tmp[1], v.tmp[2]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1], v.tmp[1]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[2], v.tmp[2]);
             dg::blas1::scal(v.tmp[1], -1);
             dg::blas1::scal(v.tmp[2], -1);//b_\perp

			 dg::blas1::pointwiseDot(v.f.velocity(1), v.f.gradN(1)[0], v.tmp2[0]);
             dg::blas1::pointwiseDot(v.f.velocity(1), v.f.gradN(1)[1], v.tmp2[1]);
             dg::blas1::pointwiseDot(v.f.density(1), v.f.gradU(1)[0], v.tmp3[0]);
             dg::blas1::pointwiseDot(v.f.density(1), v.f.gradU(1)[1], v.tmp3[1]);	
             dg::blas1::pointwiseDot(v.tmp[0], v.tmp2[1], v.tmp2[1]);
             dg::blas1::pointwiseDot(v.tmp[0], v.tmp2[2], v.tmp2[2]);
             dg::blas1::pointwiseDot(v.tmp[0], v.tmp3[1], v.tmp3[1]);
             dg::blas1::pointwiseDot(v.tmp[0], v.tmp3[2], v.tmp3[2]);	
             dg::blas1::axpby(1, v.tmp2[0], 1, v.tmp3[0]);
             dg::blas1::axpby(1, v.tmp2[1], 1, v.tmp3[1]);
             nabla.div(v.tmp3[0], v.tmp3[1], result);
                  
             dg::blas1::pointwiseDot(result, v.tmp[1], v.tmp[1]);
             dg::blas1::pointwiseDot(result, v.tmp[2], v.tmp[2]);
             nabla.div(v.tmp[1], v.tmp[2], result);      
        }
    },
    
    {"curv_term_tt", "curvature term (time integrated)", true, //FINAL
        []( DVec& result, Variables& v) {           
            dg::blas1::copy(v.f.density(0), v.tmp[0]); 
            dg::blas1::scal(v.tmp[0],-1);
            dg::blas1::axpby(v.p.tau[1], v.f.density(1), v.p.tau[0], v.tmp[0]);
            dg::blas1::pointwiseDot(v.tmp[0], v.f.curv()[1], v.tmp[1]);  
            dg::blas1::pointwiseDot(v.tmp[0], v.f.curv()[0], v.tmp[0]);
            nabla.div(v.tmp[0], v.tmp[1], result);  
        }
    },
    
     {"curvKappa_term_tt", "curvature Kappa term (time integrated)", true, //FINAL
        []( DVec& result, Variables& v) {                       
            dg::blas1::copy(v.f.density(0), v.tmp2[0]); 
            dg::blas1::scal(v.tmp2[0], -1); 
            dg::blas1::copy(v.f.density(1), v.tmp2[1]);  
            dg::blas1::pointwiseDot(v.f.velocity(0), v.tmp2[0], v.tmp2[0]);
            dg::blas1::pointwiseDot(v.f.velocity(0), v.tmp2[0], v.tmp2[0]);
            dg::blas1::pointwiseDot(v.f.velocity(1), v.tmp2[1], v.tmp2[1]);
            dg::blas1::pointwiseDot(v.f.velocity(1), v.tmp2[1], v.tmp2[1]);
            dg::blas1::axpby(v.p.mu[0], v.tmp2[0], v.p.mu[1], v.tmp2[1]); 
            dg::blas1::pointwiseDot(v.tmp2[1], v.f.curvKappa()[0], v.tmp2[0]);
            dg::blas1::pointwiseDot(v.tmp2[1], v.f.curvKappa()[1], v.tmp2[1]);               
            nabla.div(v.tmp2[0], v.tmp2[1], result);           
        }
    },
    {"elec_S_vorticity_term_tt", "Electric source vorticity (time integrated)", true, //FINAL
        []( DVec& result, Variables& v) {
             dg::blas1::pointwiseDot(v.f.density_source(1), v.f.gradP(0)[0], v.tmp[0]);
             dg::blas1::pointwiseDot(v.f.density_source(1), v.f.gradP(0)[1], v.tmp[1]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1]);   
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1]);  
             dg::tensor.multiply2d(v.f.projection(), v.tmp[0], v.tmp[1], v.tmp[0], v.tmp[1]); //to transform the vector from covariant to contravariant    
             nabla.div(v.tmp[0], v.tmp[1], result);       
        }
    },
    
    {"dielec_S_vorticity_term_tt", "Dielectric source vorticity (time integrated)", true, //FINAL
        []( DVec& result, Variables& v) {
             v.f.compute_gradSN(0,  v.tmp); 
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0]);    
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1]);    
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1]);
             dg::tensor.multiply2d(v.f.projection(), v.tmp[0], v.tmp[1], v.tmp[0], v.tmp[1]); //to transform the vector from covariant to contravariant             
             nabla.div(v.tmp[0], v.tmp[1], result);
             dg::blas1::scal(result, v.p.tau[1]);      
        }
    },
    
    {"current_perp_term_tt", "Perp gradient current term (time integrated)", true, //FINAL
        []( DVec& result, Variables& v) {
             nabla.b_cross_v (v.f.gradA()[0], v.f.gradA()[1], v.tmp[0], v.tmp[1]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0], v.tmp[0]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1], v.tmp[1]);//b_perp
                        
             dg::blas1::pointwiseDot(v.f.density(1), v.f.velocity(1), v.tmp[2]);
             dg::blas1::pointwiseDot(-1., v.f.density(0), v.f.velocity(0), 1., v.tmp[2]);
             dg::blas1::copy(v.tmp[2], v.tmp2[0]); //v.tmp2[0]=J_||
             nabla.Grad_perp_f(v.tmp[2], v.tmp[2], result);
             dg::blas1::pointwiseDot(v.tmp[0], v.tmp[2], v.tmp[0]);
             dg::blas1::pointwiseDot(v.tmp[1], result, result);
             dg::blas1::axpby(1, v.tmp[0], 1, result); //grad J_||*b_perp
             
             dg::pointwiseDot(v.f.induction(), v.f.divCurvKappa(), v.tmp[2]); 
             dg::blas1::copy(v.f.curv()[0], v.tmp[0]); 
             dg::blas1::copy(v.f.curv()[1], v.tmp[1]); 
             dg::blas1::axpby(1, v.f.curvKappa()[0], -1, v.tmp[0]);
             dg::blas1::axpby(1, v.f.curvKappa()[1], -1, v.tmp[1]);
             dg::blas1::pointwiseDot(v.tmp[0], v.f.gradA()[0], v.tmp[0]);
             dg::blas1::pointwiseDot(v.tmp[1], v.f.gradA()[1], v.tmp[1]);
             dg::blas1::axpbypgz(1, v.tmp[1], 1, v.tmp[0], 1, v.tmp[2]);  
             dg::blas1::pointwiseDot(v.tmp2[0], v.tmp[2], v.tmp[2]);
             dg::blas1::axpby(1, v.tmp[2], 1, result); 
                    
        }
    },
    
    {"elec_extra_term_tt", "Electric extra term (time integrated)", true, //FINAL
        []( DVec& result, Variables& v) {             
             nabla.b_cross_v(v.f.gradP(0)[0], v.f.gradP(0)[1], v.tmp[0], v.tmp[1]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0], v.tmp[0]); 
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1], v.tmp[1]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0], v.tmp[0]); 
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1], v.tmp[1]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0], v.tmp[0]); 
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[1], v.tmp[1]); //u_e/B^2
                        
             dg::blas1::pointwiseDot(v.tmp[0], v.f.gradN(1)[0], v.tmp2[0]);
             dg::blas1::pointwiseDot(v.tmp[1], v.f.gradN(1)[1], v.tmp2[1]);
             dg::blas1::axpby(1, v.tmp2[0], 1, v.tmp2[1]);
             nabla.div(v.tmp[0], v.tmp[1], v.tmp[0]);
             dg::blas1::pointwiseDot(v.f.density(1), v.tmp[0], v.tmp[0]);
             dg::blas1::axpby(1, v.tmp2[1], 1, v.tmp[0]);
             nabla.Grad_perp_f(v.tmp[0], v.tmp[1], v.tmp[2]);
             nabla.div(v.tmp[1], v.tmp[2], result);  
             dg::blas1::scal(result, v.p.tau[1]);          
        }
    },
    
    {"par_extra_term_tt", "Parallel extra term (time integrated)", true, //FINAL
        []( DVec& result, Variables& v) { 
             dg::blas1::pointwiseDot(v.f.velocity(1), v.f.gradN(1)[0], v.tmp[0]);
             dg::blas1::pointwiseDot(v.f.velocity(1), v.f.gradN(1)[1], v.tmp[1]);
             dg::blas1::pointwiseDot(v.f.density(1), v.f.gradU(1)[0], v.tmp2[0]);
             dg::blas1::pointwiseDot(v.f.density(1), v.f.gradU(1)[1], v.tmp2[1]);
             dg::blas1::axpby(1, v.tmp2[0], 1, v.tmp[0]);
             dg::blas1::axpby(1, v.tmp2[1], 1, v.tmp[1]); //U grad N+N grad U
                         
             nabla.b_cross_v(v.f.gradP(0)[0], v.f.gradP(0)[1], v.tmp2[0], v.tmp2[1]);
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp2[0], v.tmp2[0]); 
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp2[1], v.tmp2[1]); //b_perp
                				
			 dg::blas1::pointwiseDot(v.tmp[0], v.tmp2[0], v.tmp[0])
			 dg::blas1::pointwiseDot(v.tmp[1], v.tmp2[1], v.tmp[1])
			 dg::blas1::axpby(1, v.tmp[1], 1, v.tmp[0]);
			 dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0], v.tmp[0]);    
             dg::blas1::pointwiseDot(v.f.binv(), v.tmp[0], v.tmp[0]); 
             nabla.Grad_perp_f(v.tmp[0], v.tmp[1], v.tmp[2]);
             nabla.div(v.tmp[1], v.tmp[2], result); 
             dg::blas1::scal(result, v.p.tau[1]);         
        }
    },
    ///----------------------EXTRA RAUL ADDITION-------------------------///
        {"er", "Radial electric field", false,
        []( DVec& result, Variables& v){
			dg::blas1::scal( v.gradPsip, 1/sqrt(v.gradPsip[0]*v.gradPsip[0]+v.gradPsip[1]*v.gradPsip[1]), result);
            routines::dot( v.f.gradP(0), result, result);
        }
    },  
     {"par_J", "Parallel current", false,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(v.f.density(1), v.f.velocity(1), result);
            dg::blas1::pointwiseDot(-1., v.f.density(0), v.f.velocity(0), 1., result);
        }
    },
        /// -----------------Miscellaneous additions --------------------//
    {"vorticity", "Minus Lap_perp of electric potential", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.lapMperpP(0), result);
        }
    },
    {"apar_vorticity", "Minus Lap_perp of magnetic potential", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.lapMperpA(), result);
        }
    },
    {"dssue", "2nd parallel derivative of electron velocity", false,
        []( DVec& result, Variables& v ) {
            v.f.compute_dssU( 0, result);
        }
    },
    {"lperpinv", "Perpendicular density gradient length scale", false,
        []( DVec& result, Variables& v ) {
            const std::array<DVec, 3>& dN = v.f.gradN(0);
            dg::tensor::multiply3d( v.f.projection(), //grad_perp
                dN[0], dN[1], dN[2], v.tmp[0], v.tmp[1], v.tmp[2]);
            routines::dot(dN, v.tmp, result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
            dg::blas1::transform( result, result, dg::SQRT<double>());
        }
    },
    {"perpaligned", "Perpendicular density alignement", false,
        []( DVec& result, Variables& v ) {
            const std::array<DVec, 3>& dN = v.f.gradN(0);
            dg::tensor::multiply3d( v.f.projection(), //grad_perp
                dN[0], dN[1], dN[2], v.tmp[0], v.tmp[1], v.tmp[2]);
            routines::dot(dN, v.tmp, result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
        }
    },
    {"lparallelinv", "Parallel density gradient length scale", false,
        []( DVec& result, Variables& v ) {
            v.f.compute_dsN(0, result);
            dg::blas1::pointwiseDot ( result, result, result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
            dg::blas1::transform( result, result, dg::SQRT<double>());
        }
    },
    {"aligned", "Parallel density alignement", false,
        []( DVec& result, Variables& v ) {
            v.f.compute_dsN(0, result);
            dg::blas1::pointwiseDot ( result, result, result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
        }
    },
    /// ------------------ Correlation terms --------------------//
    {"ne2", "Square of electron density", false,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.density(0), v.f.density(0), result);
        }
    },
    {"phi2", "Square of electron potential", false,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.potential(0), v.f.potential(0), result);
        }
    },
    {"nephi", "Product of electron potential and electron density", false,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.potential(0), v.f.density(0), result);
        }
    },
    /// ------------------ Density terms ------------------------//
    {"jsneE_tt", "Radial electron particle flux: ExB contribution (Time average)", true,
        []( DVec& result, Variables& v ) {
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( result, v.f.density(0), result);
        }
    },
    {"jsneC_tt", "Radial electron particle flux: curvature contribution (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialParticleFlux( v.p.tau[0], v.p.mu[0]),
                v.f.density(0), v.f.velocity(0),
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"jsdiae_tt", "Radial electron particle flux: diamagnetic contribution (Time average)", true,
        []( DVec& result, Variables& v ) {
            // u_D Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradN(0), v.gradPsip, result);
            dg::blas1::scal( result, v.p.tau[0]);
        }
    },
    {"jsneA_tt", "Radial electron particle flux: magnetic contribution (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialParticleFlux( v.p.tau[0], v.p.mu[0]),
                v.f.density(0), v.f.velocity(0), v.f.induction(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"lneperp_tt", "Perpendicular electron diffusion (Time average)", true,
        []( DVec& result, Variables& v ) {
            v.f.compute_diffusive_lapMperpN( v.f.density(0), v.tmp[0], result);
            dg::blas1::scal( result, -v.p.nu_perp);
        }
    },
    //{"lneparallel_tt", "Parallel electron diffusion (Time average)", true,
    //    []( DVec& result, Variables& v ) {
    //        dg::blas1::pointwiseDot( v.p.nu_parallel, v.f.divb(), v.f.dsN(0),
    //                                 0., result);
    //        dg::blas1::axpby( v.p.nu_parallel, v.f.dssN(0), 1., result);
    //    }
    //},
    {"sne_tt", "Source term for electron density (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::copy( v.f.density_source(0), result);
        }
    },
    {"divnepar_tt", "Divergence of Parallel velocity term for electron density (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 1., v.f.density(0), v.f.velocity(0), v.f.divb(), 0., result);
            v.f.compute_dsU(0, v.tmp[0]);
            dg::blas1::pointwiseDot( 1., v.f.density(0),  v.tmp[0], 1., result);
            v.f.compute_dsN(0, v.tmp[0]);
            dg::blas1::pointwiseDot( 1., v.f.velocity(0), v.tmp[0], 1., result);
        }
    },
    {"jsniE_tt", "Radial ion particle flux: ExB contribution (Time average)", true,
        []( DVec& result, Variables& v ) {
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(1), v.gradPsip, result);
            dg::blas1::pointwiseDot( result, v.f.density(1), result);
        }
    },
    {"jsniC_tt", "Radial ion particle flux: curvature contribution (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialParticleFlux( v.p.tau[1], v.p.mu[1]),
                v.f.density(1), v.f.velocity(1),
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"jsdiai_tt", "Radial ion particle flux: diamagnetic contribution (Time average)", true,
        []( DVec& result, Variables& v ) {
            // u_D Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradN(1), v.gradPsip, result);
            dg::blas1::scal( result, v.p.tau[1]);
        }
    },
    {"jsniA_tt", "Radial ion particle flux: magnetic contribution (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialParticleFlux( v.p.tau[1], v.p.mu[1]),
                v.f.density(1), v.f.velocity(1), v.f.induction(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"lniperp_tt", "Perpendicular ion diffusion (Time average)", true,
        []( DVec& result, Variables& v ) {
            v.f.compute_diffusive_lapMperpN( v.f.density(1), v.tmp[0], result);
            dg::blas1::scal( result, -v.p.nu_perp);
        }
    },
    //{"lniparallel_tt", "Parallel ion diffusion (Time average)", true,
    //    []( DVec& result, Variables& v ) {
    //        dg::blas1::pointwiseDot( v.p.nu_parallel, v.f.divb(), v.f.dsN(1),
    //                                 0., result);
    //        dg::blas1::axpby( v.p.nu_parallel, v.f.dssN(1), 1., result);
    //    }
    //},
    {"sni_tt", "Source term for ion density (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::copy( v.f.density_source(1), result);
        }
    },
    {"divnipar_tt", "Divergence of Parallel velocity term in ion density (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 1., v.f.density(1), v.f.velocity(1), v.f.divb(), 0., result);
            v.f.compute_dsU(1, v.tmp[1]);
            dg::blas1::pointwiseDot( 1., v.f.density(1),  v.tmp[1], 1., result);
            v.f.compute_dsN(1, v.tmp[1]);
            dg::blas1::pointwiseDot( 1., v.f.velocity(1), v.tmp[1], 1., result);
        }
    },
    /// ------------------- Energy terms ------------------------//
    {"nelnne", "Entropy electrons", false,
        []( DVec& result, Variables& v ) {
            dg::blas1::transform( v.f.density(0), result, dg::LN<double>());
            dg::blas1::pointwiseDot( result, v.f.density(0), result);
        }
    },
    {"nilnni", "Entropy ions", false,
        []( DVec& result, Variables& v ) {
            dg::blas1::transform( v.f.density(1), result, dg::LN<double>());
            dg::blas1::pointwiseDot( v.p.tau[1], result, v.f.density(1), 0., result);
        }
    },
    {"aperp2", "Magnetic energy", false,
        []( DVec& result, Variables& v ) {
            if( v.p.beta == 0)
            {
                dg::blas1::scal( result, 0.);
            }
            else
            {
                dg::tensor::multiply3d( v.f.projection(), //grad_perp
                    v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                    v.tmp[0], v.tmp[1], v.tmp[2]);
                routines::dot( v.tmp, v.f.gradA(), result);
                dg::blas1::scal( result, 1./2./v.p.beta);
            }
        }
    },
    {"ue2", "ExB energy", false,
        []( DVec& result, Variables& v ) {
            dg::tensor::multiply3d( v.f.projection(), //grad_perp
                v.f.gradP(0)[0], v.f.gradP(0)[1], v.f.gradP(0)[2],
                v.tmp[0], v.tmp[1], v.tmp[2]);
            routines::dot( v.tmp, v.f.gradP(0), result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( 0.5, v.f.density(1), result, 0., result);
        }
    },
    {"neue2", "Parallel electron energy", false,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( -0.5*v.p.mu[0], v.f.density(0),
                v.f.velocity(0), v.f.velocity(0), 0., result);
        }
    },
    {"niui2", "Parallel ion energy", false,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 0.5*v.p.mu[1], v.f.density(1),
                v.f.velocity(1), v.f.velocity(1), 0., result);
        }
    },
    /// ------------------- Energy dissipation ----------------------//
    {"resistivity_tt", "Energy dissipation through resistivity (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::axpby( 1., v.f.velocity(1), -1., v.f.velocity(0), result);
            dg::blas1::pointwiseDot( result, v.f.density(0), result);
            dg::blas1::pointwiseDot( -v.p.eta, result, result, 0., result);
        }
    },
    {"see_tt", "Energy sink/source for electrons", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.f.density_source(0)
            );
        }
    },
    {"sei_tt", "Energy sink/source for ions", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.f.density_source(1)
            );
        }
    },
    /// ------------------ Energy flux terms ------------------------//
    {"jsee_tt", "Radial electron energy flux without magnetic contribution (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.f.gradP(0)[0], v.f.gradP(0)[1], v.f.gradP(0)[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"jseea_tt", "Radial electron energy flux: magnetic contribution (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0), v.f.induction(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"jsei_tt", "Radial ion energy flux without magnetic contribution (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.f.gradP(1)[0], v.f.gradP(1)[1], v.f.gradP(1)[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"jseia_tt", "Radial ion energy flux: magnetic contribution (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1), v.f.induction(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    /// ------------------------ Energy dissipation terms ------------------//
    {"leeperp_tt", "Perpendicular electron energy dissipation (Time average)", true,
        []( DVec& result, Variables& v ) {
            v.f.compute_diffusive_lapMperpN( v.f.density(0), result, v.tmp[0]);
            v.f.compute_diffusive_lapMperpU( v.f.velocity(0), result, v.tmp[1]);
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.tmp[0], v.tmp[1]
            );
            dg::blas1::scal( result, -v.p.nu_perp);
        }
    },
    {"leiperp_tt", "Perpendicular ion energy dissipation (Time average)", true,
        []( DVec& result, Variables& v ) {
            v.f.compute_diffusive_lapMperpN( v.f.density(1), result, v.tmp[0]);
            v.f.compute_diffusive_lapMperpU( v.f.velocity(1), result, v.tmp[1]);
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.tmp[0], v.tmp[1]
            );
            dg::blas1::scal( result, -v.p.nu_perp);
        }
    },
    {"leeparallel_tt", "Parallel electron energy dissipation (Time average)", true,
        []( DVec& result, Variables& v ) {
            //v.f.compute_lapParN( 0, v.tmp[0]);
            dg::blas1::copy(0., v.tmp[0]);
            v.f.compute_lapParU( 0, v.tmp[1]);
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.tmp[0], v.tmp[1]
            );
            dg::blas1::scal( result, v.p.nu_parallel[0]);
        }
    },
    {"leiparallel_tt", "Parallel ion energy dissipation (Time average)", true,
        []( DVec& result, Variables& v ) {
            //v.f.compute_lapParN( 1, v.tmp[0]);
            dg::blas1::copy(0., v.tmp[0]);
            v.f.compute_lapParU( 1, v.tmp[1]);
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.tmp[0], v.tmp[1]
            );
            dg::blas1::scal( result, v.p.nu_parallel[1]);
        }
    },
    /// ------------------------ Vorticity terms ---------------------------//
    {"oexbi", "ExB vorticity term with ion density", false,
        []( DVec& result, Variables& v){
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density(1), 0., result);
        }
    },
    {"oexbe", "ExB vorticity term with electron density", false,
        []( DVec& result, Variables& v){
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density(0), 0., result);
        }
    },
    {"odiai", "Diamagnetic vorticity term with ion density", false,
        []( DVec& result, Variables& v){
            routines::dot( v.f.gradN(1), v.gradPsip, result);
            dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    {"odiae", "Diamagnetic vorticity term with electron density", false,
        []( DVec& result, Variables& v){
            routines::dot( v.f.gradN(0), v.gradPsip, result);
            dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    /// --------------------- Vorticity flux terms ---------------------------//
    {"jsoexbi_tt", "ExB vorticity flux term with ion density (Time average)", true,
        []( DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // Omega_E
            routines::dot( v.f.gradP(0), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.binv(), v.f.binv(), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], v.f.density(1), 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"jsoexbe_tt", "ExB vorticity flux term with electron density (Time average)", true,
        []( DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // Omega_E
            routines::dot( v.f.gradP(0), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.binv(), v.f.binv(), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], v.f.density(0), 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"jsodiaiUE_tt", "Diamagnetic vorticity flux by ExB veloctiy term with ion density (Time average)", true,
        []( DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // Omega_D,phi
            routines::dot( v.f.gradN(1), v.gradPsip, v.tmp[0]);
            dg::blas1::scal( v.tmp[0], v.p.mu[1]*v.p.tau[1]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"jsodiaeUE_tt", "Diamagnetic vorticity flux by ExB velocity term with electron density (Time average)", true,
        []( DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // Omega_D,phi
            routines::dot( v.f.gradN(0), v.gradPsip, v.tmp[0]);
            dg::blas1::scal( v.tmp[0], v.p.mu[1]*v.p.tau[1]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"jsoexbiUD_tt", "ExB vorticity flux term by diamagnetic velocity with ion density (Time average)", true,
        []( DVec& result, Variables& v){
            // bxGradN/B Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradN(1), v.gradPsip, result);
            dg::blas1::scal( result, v.p.tau[1]);

            // m Omega_E,phi
            routines::dot( v.f.gradP(0), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], v.f.binv(), v.f.binv(), 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"jsoexbeUD_tt", "ExB vorticity flux term by diamagnetic velocity with electron density (Time average)", true,
        []( DVec& result, Variables& v){
            // bxGradN/B Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradN(0), v.gradPsip, result);
            dg::blas1::scal( result, v.p.tau[1]);

            // m Omega_E,phi
            routines::dot( v.f.gradP(0), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], v.f.binv(), v.f.binv(), 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"jsoapar_tt", "A parallel vorticity flux term (Maxwell stress) (Time average)", true,
        []( DVec& result, Variables& v){
            if( v.p.beta == 0)
                dg::blas1::scal( result, 0.);
            else
            {
                routines::jacobian( v.f.bhatgB(), v.f.gradA(), v.gradPsip, result);
                routines::dot( v.f.gradA(), v.gradPsip, v.tmp[0]);
                dg::blas1::pointwiseDot( -1./v.p.beta, result, v.tmp[0], 0., result);
            }
        }
    },
    {"jsodiaApar_tt", "A parallel diamagnetic vorticity flux term (magnetization stress) (Time average)", true,
        []( DVec& result, Variables& v){
            if( v.p.beta == 0)
                dg::blas1::scal( result, 0.);
            else
            {
                routines::dot( v.gradPsip, v.f.gradU(1), v.tmp[0]);
                routines::dot( v.gradPsip, v.f.gradN(1), v.tmp[1]);
                dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.density(1), 1., v.tmp[0], v.f.velocity(1), 0., result);

                routines::jacobian( v.f.bhatgB(), v.f.gradA(), v.gradPsip, result);
                dg::blas1::pointwiseDot( -1./2.*v.p.tau[1], result, v.tmp[0], 0., result);
            }
        }
    },
    {"jsoexbApar_tt", "A parallel ExB vorticity flux term (magnetization stress) (Time average)", true,
        []( DVec& result, Variables& v){
            if( v.p.beta == 0)
                dg::blas1::scal( result, 0.);
            else
            {
                routines::jacobian( v.f.bhatgB(), v.f.gradU(1), v.gradPsip, v.tmp[0]);
                routines::jacobian( v.f.bhatgB(), v.f.gradN(1), v.gradPsip, v.tmp[1]);
                dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.density(1), 1., v.tmp[1], v.f.velocity(1), 0., result);
                routines::dot( v.f.gradA(), v.gradPsip, v.tmp[2]);
                dg::blas1::pointwiseDot( -1./2.*v.p.tau[1], result, v.tmp[2], 0., result);
            }
        }
    },
    {"sosne_tt", "ExB vorticity source term with electron source", true,
        []( DVec& result, Variables& v){
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density_source(0), 0., result);
        }
    },
    {"sospi_tt", "Diamagnetic vorticity source term with electron source", true,
        []( DVec& result, Variables& v){
            v.f.compute_gradSN( 0, v.tmp);
            routines::dot( v.tmp, v.gradPsip, result);
            dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    {"loexbe_tt", "Vorticity dissipation term with electron Lambda", true,
        []( DVec& result, Variables& v){
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);

            v.f.compute_diffusive_lapMperpN( v.f.density(0), v.tmp[0], v.tmp[1]);
            dg::blas1::scal( v.tmp[1], -v.p.nu_perp);
            //v.f.compute_lapParN( 0, v.tmp[2]);
            //dg::blas1::scal( v.tmp[2], v.p.nu_parallel);
            dg::blas1::copy( 0., v.tmp[2]);
            dg::blas1::axpby( 1., v.tmp[1], 1., v.tmp[2]); //Lambda_ne
            dg::blas1::pointwiseDot( v.tmp[2], result, result);

            dg::blas1::scal( result, v.p.mu[1]);
        }
    },
    ///-----------------------Parallel momentum terms ------------------------//
    {"neue", "Product of electron density and velocity", false,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.density(0), v.f.velocity(0), result);
        }
    },
    {"niui", "Product of ion gyrocentre density and velocity", false,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.density(1), v.f.velocity(1), result);
        }
    },
    {"niuibphi", "Product of NiUi and covariant phi component of magnetic field unit vector", false,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 1.,
                v.f.density(1), v.f.velocity(1), v.f.bphi(), 0., result);
        }
    },
    /// --------------------- Parallel momentum flux terms ---------------------//
    {"jsparexbi_tt", "Parallel momentum radial flux by ExB velocity with electron potential (Time average)", true,
        []( DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // parallel momentum mu_iN_iU_i
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), result, 0., result);
        }
    },
    {"jsparbphiexbi_tt", "Parallel angular momentum radial flux by ExB velocity with ion density (Time average)", true,
        []( DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(1), v.gradPsip, result);

            // parallel momentum mu_iN_iU_i
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0],v.f.bphi(), 0., result);
        }
    },
    {"jspardiai_tt", "Parallel momentum radial flux by Diamagnetic velocity with ion density (Time average)", true,
        []( DVec& result, Variables& v){
            // DiaN Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradN(1), v.gradPsip, v.tmp[0]);
            // DiaU Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradU(1), v.gradPsip, v.tmp[1]);

            // Multiply everything
            dg::blas1::pointwiseDot( v.p.mu[1]*v.p.tau[1], v.tmp[0], v.f.velocity(1), v.p.mu[1]*v.p.tau[1], v.tmp[1], v.f.density(1), 0., result);
        }
    },
    {"jsparkappai_tt", "Parallel momentum radial flux by curvature velocity (Time average)", true,
        []( DVec& result, Variables& v){
            routines::dot( v.f.curvKappa(), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), v.tmp[0], 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.velocity(1), v.f.velocity(1), v.tmp[0], 0., v.tmp[1]);
            dg::blas1::axpbypgz( 2.*v.p.tau[1], v.tmp[0], +1., v.tmp[1], 0., result);
        }
    },
    {"jsparbphidiai_tt", "Parallel angular momentum radial flux by Diamagnetic velocity with ion density (Time average)", true,
        []( DVec& result, Variables& v){
            // bphi K Dot GradPsi
            routines::dot( v.f.curv(), v.gradPsip, result);
            dg::blas1::pointwiseDot( result, v.f.bphi(), result);
            // Multiply everything
            dg::blas1::pointwiseDot( v.p.mu[1]*v.p.tau[1], result, v.f.velocity(1), v.f.density(1), 0., result);
        }
    },
    {"jsparbphikappai_tt", "Parallel angular momentum radial flux by curvature velocity (Time average)", true,
        []( DVec& result, Variables& v){
            routines::dot( v.f.curvKappa(), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), v.tmp[0], 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.velocity(1), v.f.velocity(1), v.tmp[0], 0., v.tmp[1]);
            dg::blas1::axpbypgz( 2.*v.p.tau[1], v.tmp[0], +1., v.tmp[1], 0., result);
            dg::blas1::pointwiseDot( result, v.f.bphi(), result);
        }
    },
    {"jsparApar_tt", "Parallel momentum radial flux by magnetic flutter (Time average)", true,
        []( DVec& result, Variables& v){
            if( v.p.beta == 0)
            {
                dg::blas1::scal( result, 0.);
            }
            else
            {
                //b_\perp^v
                routines::jacobian( v.f.gradA() , v.f.bhatgB(), v.gradPsip, v.tmp[2]);
                dg::blas1::pointwiseDot( -v.p.mu[0], v.f.velocity(0), v.f.velocity(0), v.f.density(0),  0., v.tmp[0]);
                dg::blas1::pointwiseDot( +v.p.mu[1], v.f.velocity(1), v.f.velocity(1), v.f.density(1),  0., v.tmp[1]);
                dg::blas1::pointwiseDot( -v.p.tau[0], v.f.density(0), v.tmp[2], 0., result);
                dg::blas1::pointwiseDot( +v.p.tau[1], v.f.density(1), v.tmp[2], 1., result);
                dg::blas1::pointwiseDot( 1., v.tmp[0], v.tmp[2], 1., result);
                dg::blas1::pointwiseDot( 1., v.tmp[1], v.tmp[2], 1., result);
            }
        }
    },
    {"jsparbphiApar_tt", "Parallel angular momentum radial flux by magnetic flutter (Time average)", true,
        []( DVec& result, Variables& v){
            if( v.p.beta == 0)
            {
                dg::blas1::scal( result, 0.);
            }
            else
            {
                //b_\perp^v
                routines::jacobian( v.f.gradA() , v.f.bhatgB(), v.gradPsip, v.tmp[2]);
                dg::blas1::pointwiseDot( v.tmp[2], v.f.bphi(), v.tmp[2]);
                dg::blas1::pointwiseDot( -v.p.mu[0], v.f.velocity(0), v.f.velocity(0), v.f.density(0),  0., v.tmp[0]);
                dg::blas1::pointwiseDot( +v.p.mu[1], v.f.velocity(1), v.f.velocity(1), v.f.density(1),  0., v.tmp[1]);
                dg::blas1::pointwiseDot( -v.p.tau[0], v.f.density(0), v.tmp[2], 0., result);
                dg::blas1::pointwiseDot( +v.p.tau[1], v.f.density(1), v.tmp[2], 1., result);
                dg::blas1::pointwiseDot( 1., v.tmp[0], v.tmp[2], 1., result);
                dg::blas1::pointwiseDot( 1., v.tmp[1], v.tmp[2], 1., result);
            }
        }
    },
    /// --------------------- Parallel momentum source terms ---------------------//
    //Not so important (and probably not accurate)
    {"sparpar_tt", "Parallel Source for parallel momentum", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( v.f.velocity(1), v.f.velocity(1), v.tmp[1]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.tmp[1], v.f.divb(), 0., result);
            v.f.compute_dsU( 1, v.tmp[2]);
            dg::blas1::pointwiseDot( 0.5*v.p.mu[1], v.f.density(1),  v.f.velocity(1), v.tmp[2], 1., result);
            v.f.compute_dsN( 1, v.tmp[2]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[1], v.tmp[2], 1., result);
        }
    },
    //not so important
    {"spardivKappa_tt", "Divergence Kappa Source for parallel momentum", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( -v.p.mu[1]*v.p.tau[1], v.f.density(1), v.f.velocity(1), v.f.divCurvKappa(), 0., result);
        }
    },
    //not so important
    {"sparKappaphi_tt", "Kappa Phi Source for parallel momentum", true,
        []( DVec& result, Variables& v ) {
            routines::dot( v.f.curvKappa(), v.f.gradP(1), result);
            dg::blas1::pointwiseDot( -v.p.mu[1], v.f.density(1), v.f.velocity(1), result, 0., result);
        }
    },
    // should be zero in new implementation
    {"sparsni_tt", "Parallel momentum source by density and velocity sources", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( v.p.mu[1],
                v.f.density_source(1), v.f.velocity(1),
                v.p.mu[1], v.f.velocity_source(1), v.f.density(1), 0., result);
        }
    },
    {"sparsnibphi_tt", "Parallel angular momentum source by density and velocity sources", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( v.p.mu[1],
                v.f.density_source(1), v.f.velocity(1), v.f.bphi(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1],
                v.f.velocity_source(1), v.f.density(1), v.f.bphi(), 1., result);
        }
    },
    //should be zero
    {"lparpar_tt", "Parallel momentum dissipation by parallel diffusion", true,
        []( DVec& result, Variables& v ) {
            v.f.compute_lapParU( 1, result);
            dg::blas1::scal( result, v.p.nu_parallel[1]);
        }
    },
    {"lparperp_tt", "Parallel momentum dissipation by perp diffusion", true,
        []( DVec& result, Variables& v ) {
            v.f.compute_diffusive_lapMperpN( v.f.density(1), result, v.tmp[0]);
            v.f.compute_diffusive_lapMperpU( v.f.velocity(1), result, v.tmp[1]);
            dg::blas1::pointwiseDot( -v.p.nu_perp, v.tmp[0], v.f.velocity(1), -v.p.nu_perp, v.tmp[1], v.f.density(1), 0., result);
        }
    },
    /// --------------------- Mirror force term ---------------------------//
    {"sparmirrore_tt", "Mirror force term with electron density (Time average)", true,
        []( DVec& result, Variables& v){
            //dg::blas1::pointwiseDot( -v.p.tau[0], v.f.divb(), v.f.density(0), 0., result);
            v.f.compute_dsN(0, result);
            dg::blas1::scal( result, v.p.tau[0]);
        }
    },
    {"sparmirrorAe_tt", "Apar Mirror force term with electron density (Time average)", true,
        []( DVec& result, Variables& v){
            routines::jacobian( v.f.gradA() , v.f.bhatgB(), v.f.gradN(0), result);
            dg::blas1::scal( result, v.p.tau[0]);
        }
    },
    {"sparmirrori_tt", "Mirror force term with ion density (Time average)", true,
        []( DVec& result, Variables& v){
            //dg::blas1::pointwiseDot( v.p.tau[1], v.f.divb(), v.f.density(1), 0., result);
            v.f.compute_dsN(1, result);
            dg::blas1::scal( result, -v.p.tau[1]);
        }
    },
    //electric force balance usually well-fulfilled
    {"sparphie_tt", "Electric force in electron momentum density (Time average)", true,
        []( DVec& result, Variables& v){
            v.f.compute_dsP(0, result);
            dg::blas1::pointwiseDot( 1., result, v.f.density(0), 0., result);
        }
    },
    {"sparphiAe_tt", "Apar Electric force in electron momentum density (Time average)", true,
        []( DVec& result, Variables& v){
            routines::jacobian( v.f.gradA() , v.f.bhatgB(), v.f.gradP(0), result);
            dg::blas1::pointwiseDot( v.f.density(0), result, result);
        }
    },
    {"spardotAe_tt", "Apar Electric force in electron momentum density (Time average)", true,
        []( DVec& result, Variables& v){
            v.f.compute_dot_induction( result);
            dg::blas1::pointwiseDot( v.f.density(0), result, result);
        }
    },
    //These two should be almost the same
    {"sparphii_tt", "Electric force term in ion momentum density (Time average)", true,
        []( DVec& result, Variables& v){
            v.f.compute_dsP(1, result);
            dg::blas1::pointwiseDot( -1., result, v.f.density(1), 0., result);
        }
    },
    {"friction_tt", "Friction force in momentum density (Time average)", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::axpby( 1., v.f.velocity(1), -1., v.f.velocity(0), result);
            dg::blas1::pointwiseDot( v.p.eta, result, v.f.density(0), v.f.density(0), 0, result);
        }
    },
    /// --------------------- Lorentz force terms ---------------------------//
    {"socurve_tt", "Vorticity source term electron curvature (Time average)", true,
        []( DVec& result, Variables& v) {
            routines::dot( v.f.curv(), v.gradPsip, result);
            dg::blas1::pointwiseDot( -v.p.tau[0], v.f.density(0), result, 0., result);
        }
    },
    {"socurvi_tt", "Vorticity source term ion curvature (Time average)", true,
        []( DVec& result, Variables& v) {
            routines::dot( v.f.curv(), v.gradPsip, result);
            dg::blas1::pointwiseDot( v.p.tau[1], v.f.density(1), result, 0., result);
        }
    },
    {"socurvkappae_tt", "Vorticity source term electron kappa curvature (Time average)", true,
        []( DVec& result, Variables& v) {
            routines::dot( v.f.curvKappa(), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., v.f.density(0), v.f.velocity(0), v.f.velocity(0), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( -v.p.mu[0], v.tmp[0], result, 0., result);
        }
    },
    {"socurvkappai_tt", "Vorticity source term ion kappa curvature (Time average)", true,
        []( DVec& result, Variables& v) {
            routines::dot( v.f.curvKappa(), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., v.f.density(1), v.f.velocity(1), v.f.velocity(1), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], result, 0., result);
        }
    },
    /// --------------------- Zonal flow energy terms------------------------//
    {"nei0", "inertial factor", false,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( v.f.density(0), v.hoo, result);
        }
    },
    {"snei0_tt", "inertial factor source", true,
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( v.f.density_source(0), v.hoo, result);
        }
    },
};

///%%%%%%%%%%%%%%%%%%%%%%%%%%END DIAGNOSTICS LIST%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
///%%%%%%%%%%%%%%%%%%%%%%%%%%END DIAGNOSTICS LIST%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
///%%%%%%%%%%%%%%%%%%%%%%%%%%END DIAGNOSTICS LIST%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
std::vector<Record> restart3d_list = {
    {"restart_electrons", "electron density", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(0), result);
        }
    },
    {"restart_ions", "ion density", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(1), result);
        }
    },
    {"restart_Ue", "parallel electron velocity", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(0), result);
        }
    },
    {"restart_Ui", "parallel ion velocity", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(1), result);
        }
    },
    {"restart_induction", "parallel magnetic induction", false,
        []( DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.induction(), result);
        }
    }
};
// These two lists signify the quantities involved in accuracy computation
std::vector<std::string> energies = { "nelnne", "nilnni", "aperp2", "ue2","neue2","niui2"};
std::vector<std::string> energy_diff = { "resistivity_tt", "leeperp_tt", "leiperp_tt", "leeparallel_tt", "leiparallel_tt", "see_tt", "sei_tt"};

template<class Container>
void slice_vector3d( const Container& transfer, Container& transfer2d, size_t local_size2d)
{
#ifdef FELTOR_MPI
    thrust::copy(
        transfer.data().begin(),
        transfer.data().begin() + local_size2d,
        transfer2d.data().begin()
    );
#else
    thrust::copy(
        transfer.begin(),
        transfer.begin() + local_size2d,
        transfer2d.begin()
    );
#endif
}
}//namespace feltor
