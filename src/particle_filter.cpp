/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

const float INITIAL_WEIGHT = 1.0; 
const float EPS = 0.001;  	

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	static default_random_engine gen;
    num_particles = 100; // init number of particles to use
	
	// Create normal distributions for x, y and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	// Resize the `particles` vector to fit desired number of particles
	particles.resize(num_particles); 
	weights.resize(num_particles);

	for (int i = 0; i < num_particles; i++){
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = INITIAL_WEIGHT;
		weights[i] = INITIAL_WEIGHT;
	}	
	
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	// Some constants to save computation power
	const double vel_d_t = velocity * delta_t;
	const double yaw_d_t = yaw_rate * delta_t;
	const double vel_yaw = velocity/yaw_rate;
	
	static default_random_engine gen;
    normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);
	
	for (int i = 0; i < num_particles; i++){
		
        if (fabs(yaw_rate) < EPS){
			// when yaw_rate is too small, consider it didn't change
            particles[i].x += vel_d_t * cos(particles[i].theta);
            particles[i].y += vel_d_t * sin(particles[i].theta);
        }
        else{
            double theta_new = particles[i].theta + yaw_d_t;
            particles[i].x += vel_yaw * (sin(theta_new) - sin(particles[i].theta));
            particles[i].y += vel_yaw * (-cos(theta_new) + cos(particles[i].theta));
            particles[i].theta = theta_new;
        }
		
        // Add random Gaussian noise
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	const double BIG_NUMBER = 10000000.0;
	for (int i = 0; i < observations.size(); i++) {
		int current_j;
		double current_smallest_error = BIG_NUMBER;
		for (int j = 0; j < predicted.size(); j++) {
		  const double dx = predicted[j].x - observations[i].x;
		  const double dy = predicted[j].y - observations[i].y;
		  const double error = dx * dx + dy * dy;
		  if (error < current_smallest_error) {
			current_j = j;
			current_smallest_error = error;
		  }
		}
		observations[i].id = current_j;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	// constants used later for calculating the new weights
 	const double stdx = std_landmark[0];
 	const double stdy = std_landmark[1];
 	const double na = 0.5 / (stdx * stdx);
 	const double nb = 0.5 / (stdy * stdy);
 	const double d = sqrt( 2.0 * M_PI * stdx * stdy);

 	for (int  i = 0; i < num_particles; i++) {

    	const double px = this->particles[i].x;
    	const double py = this->particles[i].y;
    	const double ptheta = this->particles[i].theta;

    	vector<LandmarkObs> landmarks_in_range;
    	vector<LandmarkObs> map_observations;

	   /**************************************************************
		* STEP 1:
		* transform each observations to map coordinates
		* assume observations are made in the particle's perspective
		**************************************************************/
    	for (int j = 0; j < observations.size(); j++){

			const int oid = observations[j].id;
			const double ox = observations[j].x;
			const double oy = observations[j].y;
			const double transformed_x = px + ox * cos(ptheta) - oy * sin(ptheta);
			const double transformed_y = py + oy * cos(ptheta) + ox * sin(ptheta);

			LandmarkObs observation = {
       			oid,
        		transformed_x,
        		transformed_y
      		};

      		map_observations.push_back(observation);
    	}

	   /**************************************************************
		* STEP 2:
		* Find map landmarks within the sensor range
		**************************************************************/
    	for (int j = 0;  j < map_landmarks.landmark_list.size(); j++) {

			const int mid = map_landmarks.landmark_list[j].id_i;
      		const double mx = map_landmarks.landmark_list[j].x_f;
      		const double my = map_landmarks.landmark_list[j].y_f;

      		const double dx = mx - px;
      		const double dy = my - py;
      		const double error = sqrt(dx * dx + dy * dy);

      		if (error < sensor_range) {
        		LandmarkObs landmark_in_range = {
          			mid,
          			mx,
          			my
         		};

      	  		landmarks_in_range.push_back(landmark_in_range);
     		}
    	}

	  /**************************************************************
	   * STEP 3:
	   * Associate landmark in range (id) to landmark observations
	   * this function modifies std::vector<LandmarkObs> observations
	   * NOTE: - all landmarks are in map coordinates
	   *       - all observations are in map coordinates
	   **************************************************************/
   		this->dataAssociation(landmarks_in_range, map_observations);

	   /**************************************************************
		* STEP 4:
		* Compare each observation (by actual vehicle) to corresponding
		* observation by the particle (landmark_in_range)
		* update the particle weight based on this
		**************************************************************/
   	 	double w = INITIAL_WEIGHT;

    	for (int j = 0; j < map_observations.size(); j++){

			const int oid = map_observations[j].id;
      		const double ox = map_observations[j].x;
      		const double oy = map_observations[j].y;

      		const double predicted_x = landmarks_in_range[oid].x;
     	 	const double predicted_y = landmarks_in_range[oid].y;

      		const double dx = ox - predicted_x;
      		const double dy = oy - predicted_y;

      		const double a = na * dx * dx;
      		const double b = nb * dy * dy;
      		const double r = exp(-(a + b)) / d;
      		w *= r;
    	}

    	this->particles[i].weight = w;
    	this->weights[i] = w;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	static default_random_engine gen;
 
    discrete_distribution<> dist_particles(weights.begin(), weights.end());
    vector<Particle> new_particles;
    new_particles.resize(num_particles);
    for (int i = 0; i < num_particles; i++) {
        new_particles[i] = particles[dist_particles(gen)];
    }
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
