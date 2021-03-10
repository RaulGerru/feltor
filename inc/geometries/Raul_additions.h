#pragma once
#include <functional>
#include "dg/backend/memory.h"
#include "dg/topology/geometry.h"

namespace dg
{
namespace geo
{
///@addtogroup fluxfunctions


struct Grid_cutter : public aCylindricalFunctor<Grid_cutter>
{
	/**
    * @brief Cuts a 2D X-grid from a certain central poloidal position (horizontal line in the X-grid) to a range around it (a width in the y direction around the center). 
    *
    * \f[ f(zeta,eta)= \begin{cases}
	*1 \text{ if } eta_0-eta_size/2< eta < eta_0+eta_size/2 \\
	*0 \text{ else }
	*\end{cases}
	*\f]
    * 
    * 
    * @brief <tt> Grid_cutter( eta_0, eta_size) </tt>
    * @tparam double
    * @param eta_0 (center of the range you want, in radians)
    * @tparam double
    * @param eta_size (width of the poloidal range you want to cut, in degrees)
    * 
    * @note How to use it? dg::evaluate(dg::geo::Grid_cutter(eta, Range), GridX2d()); After you have it, you usually pointwise this function to a matrix of data to apply the cut to your data: dg::blas1::pointwiseDot(data, dg::evaluate(dg::geo::Grid_cutter(eta, Range), GridX2d()), cutted_data);
    */
	

    Grid_cutter(double eta_0, double eta_size): eta0(eta_0), etasize(eta_size){} //eta_0 is in radians and eta_size is in degrees
    
    double do_compute(double zeta, double eta) const { //go over all the point in the grid to return 1 or 0
	double eta_up_lim=eta0+etasize*M_PI/(2*180); //Define the upper and down limits of the cut  !!!IF THIS COULD BE DONE OUT OF THE LOOP, IT WOULD MAKE EVERYTHING EASIER!!! NO SENSE TO DO IT IN  EVERY ITERATION.
    double eta_down_lim=eta0-etasize*M_PI/(2*180);
    
    //As the grid goes from 0 to 2pi, we need to check that the lower limit is not under 0 or the higher over 2pi.
    // If that happens, we need to translate the limits to our range and change the conditions of our loops
    if (eta_up_lim>2*M_PI) {		
		eta_up_lim+=-2*M_PI;
        if( (eta<eta_up_lim || eta>eta_down_lim))
            return 1;
        return 0;
	}
    if (eta_down_lim<0)  {
		eta_down_lim+=2*M_PI;
        if( (eta<eta_up_lim || eta>eta_down_lim))
            return 1;
        return 0;   
	}
    else
    {
        if( eta<eta_up_lim && eta>eta_down_lim)
            return 1;
        return 0;
	}
    }
    private:
    double eta0, etasize;
}; 

struct radial_cut
{
	radial_cut(RealCurvilinearGridX2d<double> gridX2d): m_g(gridX2d){}
	
	HVec cut(const HVec F, const double zeta_def){ //This functions takes a 2D object in the Xgrid plane at a define radial position and saves it in a 1D variable with eta dependence.
	dg::Grid1d g1d_out_eta(m_g.y0(), m_g.y1(), m_g.n(), m_g.Ny(), dg::DIR_NEU); 
	m_conv_LCFS_F=dg::evaluate( dg::zero, g1d_out_eta);
	unsigned int zeta_cut=round(((zeta_def-m_g.x0())/m_g.lx())*m_g.Nx()*m_g.n())-1;

	for (unsigned int eta=0; eta<m_g.n()*m_g.Ny(); eta++) 
	{m_conv_LCFS_F[eta]=F[eta*m_g.n()*m_g.Nx()+zeta_cut];};
	return m_conv_LCFS_F;	
	}
	
	private:
	RealCurvilinearGridX2d<double> m_g;
	HVec m_conv_LCFS_F;
	
};


template<class Geometry, class Matrix, class Container>                                                                                                                            
struct Nablas 
{
	using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    
	Nablas(const Geometry& geom3d): m_g(geom3d) {
		dg::blas2::transfer( dg::create::dx( m_g, dg::DIR, dg::centered), m_dR); //Derivative in R direction
		dg::blas2::transfer( dg::create::dy( m_g, dg::DIR, dg::centered), m_dZ);
		m_vol=dg::tensor::volume(m_g.metric());
		m_weights=dg::create::volume(m_g);
		m_tmp=m_tmp2=m_tmp3=m_tmp4=m_weights;
		m_metric=m_g.metric();
		
		auto bhat = dg::geo::createBHat( mag);
		bhat = dg::geo::createEPhi(+1);
		if( p.curvmode == "true")
        bhat = dg::geo::createBHat(mag);
		else if( m_reversed_field)
        bhat = dg::geo::createEPhi(-1);
		m_hh = dg::geo::createProjectionTensor( bhat, g);
		} 
	
	template<class Container>
	void Grad_perp_f(const Container& f, Container& grad_R, Container& grad_Z) { //f the input scalar and c the vector field output
	dg::blas2::symv( m_dR, f, grad_R);
	dg::blas2::symv( m_dZ, f, grad_Z); //OUTPUT: COVARIANT
	//dg::tensor::multiply2d(m_metric, grad_R, grad_Z, grad_R, grad_Z) //IF ACTIVE OUTPUT: CONTRAVARIANT
	}		
	
	template<class Container, class Container2 >		
	void div (Container& v_R, Container& v_Z, Container& F){ //INPUT: CONTRAVARIANT
	dg::blas1::pointwiseDivide(v_R, m_vol, m_tmp);
	dg::blas1::pointwiseDivide(v_Z, m_vol, m_tmp2); 
	dg::blas2::symv( m_dR, m_tmp, m_tmp3); 
	dg::blas2::symv( m_dZ, m_tmp2, m_tmp4);
	dg::blas1::axpby(1, m_tmp3, 1, m_tmp4);
	dg::blas1::pointwiseDot(m_vol, m_tmp4,F);	
	
}

	template<class Container>
	void v_dot_nabla (Container& v_R, Container& v_Z, Container& f, Container& F){ //INPUT: COVARIANT
	dg::blas2::symv( m_dR, f, m_tmp);
	dg::blas2::symv( m_dZ, f, m_tmp2);
	dg::tensor::multiply2d(m_hh, m_tmp, m_tmp2, m_tmp3, m_tmp4) //WE MAKE THE GRADIENT CONTRAVARIANT
	dg::blas1::pointwiseDot(v_R, m_tmp3, m_tmp);
	dg::blas1::pointwiseDot(v_Z, m_tmp4, F);
	dg::blas1::axpby(1, m_tmp, 1, F);
	}	
	
	void b_cross_v (Container& v_R_o, Container& v_Z_o, Container& v_R_f, Container& v_Z_f){ //INPUT: COVARIANT
	dg::tensor::multiply2d(m_hh, v_R_o, v_Z_o, m_tmp, m_tmp2); //to transform the vector from covariant to contravariant
    dg::blas1::scal(-1, m_tmp2);
    dg::blas1::pointwiseDot(m_vol, m_tmp2, v_R_f);       
	dg::blas1::pointwiseDot(m_vol, m_tmp, v_Z_f); //OUTPUT: CONTRAVARIANT
	}

	private:
	Geometry m_g;
	dg::geo::TokamakMagneticField m_mag;
    dg::SparseTensor<Container > m_metric, m_hh;
	Matrix m_dR;
	Matrix m_dZ;
	Matrix m_dP;
	Container m_vol;
	Container m_weights;
    Container m_tmp, m_tmp2, m_tmp3, m_tmp4; 
};
};//namespace geo
}//namespace dg



