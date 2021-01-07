from sympl import Stepper, get_constant, initialize_numpy_arrays_with_properties
import numpy as np
# from scipy.interpolate import CubicSpline
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import integrate
from scipy.optimize import newton

class SrikanthSeaIce(Stepper):
    """
    1-d snow-ice energy balance model.
    """

    input_properties = {
        'downwelling_longwave_flux_in_air': {
            'dims': ['*', 'interface_levels'],
            'units': 'W m^-2',
        },
        'downwelling_shortwave_flux_in_air': {
            'dims': ['*', 'interface_levels'],
            'units': 'W m^-2',
        },
        'upwelling_longwave_flux_in_air': {
            'dims': ['*', 'interface_levels'],
            'units': 'W m^-2',
        },
        'upwelling_shortwave_flux_in_air': {
            'dims': ['*', 'interface_levels'],
            'units': 'W m^-2',
        },
        'surface_upward_latent_heat_flux': {
            'dims': ['*'],
            'units': 'W m^-2',
        },
        'surface_upward_sensible_heat_flux': {
            'dims': ['*'],
            'units': 'W m^-2',
        },
        'air_temperature': {
            'dims': ['mid_levels', '*'],
            'units': 'degK',
        },
        'air_pressure': {
            'dims': ['mid_levels', '*'],
            'units': 'Pa',
        },
        'air_pressure_on_interface_levels': {
            'dims': ['interface_levels', '*'],
            'units': 'Pa',
        },
        'specific_humidity': {
            'dims': ['mid_levels', '*'],
            'units': 'kg/kg',
        },
        'surface_specific_humidity': {
            'dims': ['*'],
            'units': 'kg/kg',
        },
        'area_type': {
            'dims': ['*'],
            'units': 'dimensionless',
        },
        'surface_temperature': {
            'dims': ['*'],
            'units': 'degK',
        },
        'northward_wind': {
            'dims': ['mid_levels', '*'],
            'units': 'm s^-1',
        },
        'eastward_wind': {
            'dims': ['mid_levels', '*'],
            'units': 'm s^-1',
        },
        'sea_ice_thickness': {
            'dims': ['*'],
            'units':  'm^-1'
        },
        'open_water_temperature': {
            'dims': ['*'],
            'units': 'degK',
        },
        'ocean_mixed_layer_thickness': {
            'dims': ['*'],
            'units': 'm',
        },
    }

    output_properties = {
        'surface_temperature': {
            'dims': ['*'],
            'units': 'degK',
        },
        'sea_surface_temperature': {
            'dims': ['*'],
            'units': 'degK',
        },
        'surface_upward_latent_heat_flux': {
            'dims': ['*'],
            'units': 'W m^-2',
        },
        'surface_upward_sensible_heat_flux': {
            'dims': ['*'],
            'units': 'W m^-2',
        },
        'specific_humidity': {
            'dims': ['mid_levels', '*'],
            'units': 'kg/kg',
        },
        'surface_specific_humidity': {
            'dims': ['*'],
            'units': 'kg/kg',
        },
        'open_water_temperature': {
            'dims': ['*'],
            'units': 'degK',
        },
    }

    diagnostic_properties = {
        'surface_albedo_for_direct_shortwave': {
            'dims': ['*'],
            'units': 'dimensionless'
        },
        'surface_albedo_for_diffuse_shortwave': {
            'dims': ['*'],
            'units': 'dimensionless'
        },
        'surface_albedo_for_direct_near_infrared': {
            'dims': ['*'],
            'units': 'dimensionless'
        },
        'surface_albedo_for_diffuse_near_infrared': {
            'dims': ['*'],
            'units': 'dimensionless'
        },
    }

    def __init__(self, ocean_heat_flux=2, fractional_sea_ice_export_rate=0.1/(365*24*3600), **kwargs):
        """
        Args:
            maximum_snow_ice_height (float):
                The maximum combined height of snow and ice handled by the model in :math:`m`.
            levels (int):
                The number of levels on which temperature must be output.
	        sea_water_transition (bool):
	            If True change area_type to sea when ice thickness is 0 for sea_ice, and allow sea to freeze to sea_water
        """
        self._ocean_heat_flux = ocean_heat_flux #units [W m^-2]
        self._fractional_sea_ice_export_rate = fractional_sea_ice_export_rate #units
        self._update_constants()
        super(SrikanthSeaIce, self).__init__(**kwargs)

    def _update_constants(self):
        self._Kice = get_constant('thermal_conductivity_of_solid_phase_as_ice', 'W/m/degK')
        self._Ksnow = get_constant('thermal_conductivity_of_solid_phase_as_snow', 'W/m/degK')
        self._rho_ice = get_constant('density_of_solid_phase_as_ice', 'kg/m^3')
        self._C_ice = get_constant('heat_capacity_of_solid_phase_as_ice', 'J/kg/degK')
        self._rho_snow = get_constant('density_of_solid_phase_as_snow', 'kg/m^3')
        self._C_snow = get_constant('heat_capacity_of_solid_phase_as_snow', 'J/kg/degK')
        self._C_air = 1004.64 #heat capacity of air, inexact - it will vary, units J kg^-1 K^-1
        self._Lf = get_constant('latent_heat_of_fusion', 'J/kg')
        self._Ls = 2.834e6 #latent heat of sublimation of water at 0degC, units J/kg
        self._melting_temperature = get_constant('freezing_temperature_of_liquid_phase', 'degK')
        self._sigma = get_constant('stefan_boltzmann_constant','W m^-2 K^-4')
        self._R = get_constant('gas_constant_of_dry_air', 'J kg^-1 K^-1')
        self._CE = 1e-3 #bulk transfer coefficient for latent heat at neutral stability, approximate value from Andreas (1986)
        self._CH = 1e-3 #bulk transfer coefficient for sensible heat at neutral stability, approximate value from Andreas (1986)
        self._inverse_beer_extinction_coefficient_of_sea_ice = 0.67 #units m^-1
        self._mixed_layer_heat_capacity = 4e6 #units J m^-3 K^-1

    def array_call(self, raw_state, timestep):
        self._update_constants()

        num_cols = raw_state['area_type'].shape[0]

        #calculate surface turbulent fluxes according to Andreas (1986)
        wind_speed = (raw_state['northward_wind'][0]**2 + raw_state['eastward_wind'][0]**2)**0.5
        first_level_air_density = raw_state['air_pressure'][0]/(self._R*raw_state['air_temperature'][0])

        raw_state['surface_upward_sensible_heat_flux'] = first_level_air_density*self._C_air*self._CH*wind_speed*(raw_state['surface_temperature']-raw_state['air_temperature'][0])
        raw_state['surface_upward_latent_heat_flux'] = self._Ls*self._CE*wind_speed*(raw_state['surface_specific_humidity']-raw_state['specific_humidity'][0])*first_level_air_density

        net_heat_flux = (
            raw_state['downwelling_shortwave_flux_in_air'][:, 0] +
            raw_state['downwelling_longwave_flux_in_air'][:, 0] -
            raw_state['upwelling_shortwave_flux_in_air'][:, 0] -
            raw_state['upwelling_longwave_flux_in_air'][:, 0] -
            raw_state['surface_upward_sensible_heat_flux'] -
            raw_state['surface_upward_latent_heat_flux'] +
            self._ocean_heat_flux
        )

        outputs = initialize_numpy_arrays_with_properties(
            self.output_properties, raw_state, self.input_properties
        )

        diagnostics = initialize_numpy_arrays_with_properties(
            self.diagnostic_properties, raw_state, self.input_properties
        )

        # Copy input values
        outputs['surface_temperature'][:] = raw_state['surface_temperature']
        outputs['sea_ice_thickness'][:] = raw_state['sea_ice_thickness']

        for col in range(num_cols):
            area_type = raw_state['area_type'][col].astype(str)

            thickness_distribution = raw_state['sea_ice_thickness_distribution'][:, col]
            dimensional_thickness_coordinates = raw_state['dimensional_sea_ice_thickness_coordinates'][:, col]
            number_of_thickness_levels = thickness_distribution.shape[0]

            open_water_probability = raw_state['open_water_probability'][col]
            open_water_temperature = raw_state['open_water_temperature'][col]

            #non dimensionalise thickness coordinates
            thickness_coordinates = dimensional_thickness_coordinates/self._thickness_scale
            thickness_interval = float((np.max(thickness_coordinates.flatten())-np.min(thickness_coordinates.flatten()))/(number_of_thickness_levels-1))

            albedo_open_water = 0.2
            albedo_thickest_ice = 0.68
            albedo_sea_ice_thickness_distribution = ((albedo_thickest_ice + albedo_open_water)/2
                                                     + (albedo_open_water - albedo_thickest_ice)/2 *
                                                     np.tanh(
                                                         -dimensional_thickness_coordinates/self._inverse_beer_extinction_coefficient_of_sea_ice))

            #import old growth rate and calculate new growth rate
            growth_rate_old = raw_state['sea_ice_growth_rate'][col]
            growth_rate = (-net_heat_flux)/(self._rho_ice*self._Lf) - self._fractional_sea_ice_export_rate
            outputs['sea_ice_growth_rate'][:,col] = growth_rate

            if self._allow_open_water:
                new_thickness_distribution, new_open_water_probability = self.thickness_distribution_stepper_with_open_water(thickness_distribution,
                                                                                 growth_rate,
                                                                                 growth_rate_old,
                                                                                 thickness_coordinates,
                                                                                 thickness_interval,
                                                                                 timestep.total_seconds(),
                                                                                 self._k1,
                                                                                 self._tau,
                                                                                 self._k2,
                                                                                 self._allow_open_water,
                                                                                 open_water_probability)
            else:
                new_thickness_distribution = self.thickness_distribution_stepper_without_open_water(thickness_distribution,
                                                                        growth_rate,
                                                                        growth_rate_old,
                                                                        thickness_coordinates,
                                                                        thickness_interval,
                                                                        timestep.total_seconds(),
                                                                        self._k1,
                                                                        self._tau,
                                                                        self._k2)

            outputs['sea_ice_thickness_distribution'][:,col] = new_thickness_distribution
            outputs['open_water_probability'][col] = new_open_water_probability

            #calculate surface temp using independent linearised outgoing LW flux
            #surface_temperature_for_thickness = (net_heat_flux+raw_state['upwelling_longwave_flux_in_air'][:, 0]-self._sigma0 + 273*self._Kice/dimensional_thickness_coordinates)/(self._sigmaT+self._Kice/dimensional_thickness_coordinates)
            def surface_temperature_function(T):
                return self._sigma*T**4 - net_heat_flux - raw_state['upwelling_longwave_flux_in_air'][:, 0] - raw_state['surface_upward_sensible_heat_flux'] + first_level_air_density*self._C_air*self._CH*wind_speed*(T-raw_state['air_temperature'][0,:]) + (self._Kice/dimensional_thickness_coordinates)*(T-273)
            surface_temperature_for_thickness = newton(surface_temperature_function, np.ones(np.size(dimensional_thickness_coordinates))*raw_state['surface_temperature'])
            surface_temperature_for_thickness[surface_temperature_for_thickness>273]=273
            surface_temperature_for_thickness[surface_temperature_for_thickness<0]=0

            open_water_temperature += (net_heat_flux +
                                       raw_state['surface_upward_latent_heat_flux'])*timestep.total_seconds()/(self._mixed_layer_heat_capacity*raw_state['ocean_mixed_layer_thickness'])

            mean_ice_surface_temperature = np.dot(new_thickness_distribution,surface_temperature_for_thickness)*thickness_interval
            mean_surface_temperature = np.dot(new_thickness_distribution,surface_temperature_for_thickness)*thickness_interval + new_open_water_probability*open_water_temperature
            outputs['open_water_temperature'] = open_water_temperature
            outputs['surface_temperature'][col] = mean_surface_temperature

            outputs['surface_upward_sensible_heat_flux'][col] = first_level_air_density*self._C_air*self._CH*wind_speed*(raw_state['surface_temperature']-raw_state['air_temperature'][0])
            outputs['surface_upward_latent_heat_flux'][col] = self._Ls*self._CE*wind_speed*(raw_state['surface_specific_humidity']-raw_state['specific_humidity'][0])*first_level_air_density

            raw_state['specific_humidity'][0] += self._CE * wind_speed * (raw_state['surface_specific_humidity'] - raw_state['specific_humidity'][0]) * first_level_air_density * 9.81 / np.abs(raw_state['air_pressure_on_interface_levels'][0] - raw_state['air_pressure_on_interface_levels'][1])
            raw_state['surface_specific_humidity'] -= self._CE * wind_speed * (raw_state['surface_specific_humidity'] - raw_state['specific_humidity'][0]) * first_level_air_density * 9.81 / np.abs(raw_state['air_pressure_on_interface_levels'][0] - raw_state['air_pressure_on_interface_levels'][1])

            raw_state['air_temperature'][0] += self._CH*wind_speed*(raw_state['surface_temperature']-raw_state['air_temperature'][0]) * first_level_air_density * 9.81 / np.abs(raw_state['air_pressure_on_interface_levels'][0] - raw_state['air_pressure_on_interface_levels'][1])
            raw_state['surface_temperature'] -= self._CH*wind_speed*(raw_state['surface_temperature']-raw_state['air_temperature'][0]) * first_level_air_density * 9.81 / np.abs(raw_state['air_pressure_on_interface_levels'][0] - raw_state['air_pressure_on_interface_levels'][1])

            outputs['specific_humidity'][:] = raw_state['specific_humidity']

            if area_type == 'sea_ice':
                #column surface albedo is set to the mean of the open water and ice albedo
                surface_albedo_for_column = np.dot(new_thickness_distribution,albedo_sea_ice_thickness_distribution)*thickness_interval \
                                            + new_open_water_probability*albedo_open_water
                #the albedo should be 0.5
                diagnostics['surface_albedo_for_direct_shortwave'][col] = surface_albedo_for_column
                diagnostics['surface_albedo_for_diffuse_shortwave'][col] = surface_albedo_for_column
                diagnostics['surface_albedo_for_direct_near_infrared'][col] = surface_albedo_for_column
                diagnostics['surface_albedo_for_diffuse_near_infrared'][col] = surface_albedo_for_column

        return diagnostics, outputs

    def thickness_distribution_stepper_with_open_water(self, thickness_distribution, growth_rate, growth_rate_old, thickness_coordinates,
                                       thickness_interval, timestep, k1, tau, k2, allow_open_water, open_water_probability, duration_of_year=30):
        #define phi for use within function
        def phi(growth_rate, k1=k1, tau=tau):
            return k1 - tau * growth_rate

        #non dimensionalise timestep
        timestep = timestep/(365*24*3600)*duration_of_year

        c_1 = timestep/(4*thickness_interval)
        c_2 = (k2*timestep)/(2*thickness_interval**2)

        # construct matrices for solving Crank-Nicolson equation
        A_diagonals = [c_1 * phi(growth_rate) - c_2,
                       1 + 2 * c_2,
                       -(c_1 * phi(growth_rate) + c_2)]
        A = sparse.diags(A_diagonals, [-1, 0, 1], (200, 200), format="csr")

        B_diagonals = [c_2 - c_1 * phi(growth_rate_old),
                       1 - 2 * c_2,
                       c_2 + c_1 * phi(growth_rate_old)]
        B = sparse.diags(B_diagonals, [-1, 0, 1], (200, 200), format="csr")

        # apply boundary conditions that g=0 at either h=0 and large h
        A[0, 0] = 1
        A[0, 1] = 0
        A[-1, -1] = 1
        A[-1, -2] = 0
        B[0, 0] = 1 + (timestep/0.1) * phi(growth_rate[0]) - k2 * timestep / (0.1 * thickness_interval)
        B[0, 1] = k2 * timestep / (0.1 * thickness_interval)
        B[-1, -1] = 0
        B[-1, -2] = 0

        rhs = B * thickness_distribution
        thickness_distribution = sparse.linalg.spsolve(A, rhs)

        flux_at_zero = -k2 * (thickness_distribution[1] - thickness_distribution[0]) / thickness_interval - phi(
            growth_rate[0]) * thickness_distribution[0]
        open_water_probability += -flux_at_zero*timestep

        # normalize thickness distribution
        thickness_distribution = thickness_distribution/(
                    integrate.trapz(thickness_distribution, thickness_coordinates) + open_water_probability)
        open_water_probability = open_water_probability/(
                    integrate.trapz(thickness_distribution, thickness_coordinates) + open_water_probability)

        return thickness_distribution, open_water_probability

    def thickness_distribution_stepper_without_open_water(self, thickness_distribution, growth_rate, growth_rate_old, thickness_coordinates,
                                       thickness_interval, timestep, k1, tau, k2, duration_of_year=30):
        #define phi for use within function
        def phi(growth_rate, k1=k1, tau=tau):
            return k1 - tau * growth_rate

        #non dimensionalise timestep
        timestep = timestep/(365*24*3600)*duration_of_year

        c_1 = timestep/(4*thickness_interval)
        c_2 = (k2*timestep)/(2*thickness_interval**2)

        # construct matrices for solving Crank-Nicolson equation
        A_diagonals = [c_1 * phi(growth_rate) - c_2,
                       1 + 2 * c_2,
                       -(c_1 * phi(growth_rate) + c_2)]
        A = sparse.diags(A_diagonals, [-1, 0, 1], (200, 200), format="csr")

        B_diagonals = [c_2 - c_1 * phi(growth_rate_old),
                       1 - 2 * c_2,
                       c_2 + c_1 * phi(growth_rate_old)]
        B = sparse.diags(B_diagonals, [-1, 0, 1], (200, 200), format="csr")

        # apply boundary conditions that g=0 at either h=0 and large h
        A[0, 0] = 1
        A[0, 1] = 0
        A[-1, -1] = 1
        A[-1, -2] = 0
        B[0, 0] = 0
        B[0, 1] = 0
        B[-1, -1] = 0
        B[-1, -2] = 0

        rhs = B * thickness_distribution
        thickness_distribution = sparse.linalg.spsolve(A, rhs)

        # normalize thickness distribution
        thickness_distribution = thickness_distribution / integrate.trapz(thickness_distribution, thickness_coordinates)

        return thickness_distribution

    def calculate_new_ice_temperature(self, rho, specific_heat, kappa,
                                      temp_profile, dt, dz,
                                      num_layers, surf_temperature, net_flux,
                                      soil_temperature=None):

        r = np.zeros(num_layers)
        a_sub = np.zeros(num_layers)
        a_sup = np.zeros(num_layers)

        K_interface = 0.5*(kappa[:-1] + kappa[1:])
        K_mid = kappa

        heat_capacity = rho * specific_heat
        heat_capacity_int = 0.5*(heat_capacity[:-1] + heat_capacity[1:])

        mu_inv_int = dt / (heat_capacity_int * 2 * dz * dz)

        r[1:-1] = K_interface*mu_inv_int

        dp = (1 + 2*r)
        dm = (1 - 2*r)

        a_sub[:-2] = -mu_inv_int*K_mid[:-1]
        a_sup[2:] = -mu_inv_int*K_mid[1:]

        mat_lhs = sparse.spdiags([a_sub, dp, a_sup], [-1, 0, 1], num_layers, num_layers, format='csc')

        mat_rhs = sparse.spdiags([-a_sub, dm, -a_sup], [-1, 0, 1], num_layers, num_layers, format='csc')

        rhs = mat_rhs * temp_profile

        # Set flux condition if temperature is below melting point,
        # and dirichlet condition above melting point
        if surf_temperature < self._melting_temperature - self._epsilon:
            mat_lhs[-1, -1] = -1
            mat_lhs[-1, -2] = 1
            rhs[-1] = -net_flux*dz/K_mid[-1]
        else:
            mat_lhs[-1, -1] = 1
            mat_lhs[-1, -2] = 0
            rhs[-1] = self._melting_temperature

        mat_lhs[0, 0] = 1
        mat_lhs[0, 1] = 0
        if soil_temperature is None:
            rhs[0] = self._melting_temperature
        else:
            rhs[0] = soil_temperature

        return spsolve(mat_lhs, rhs)
