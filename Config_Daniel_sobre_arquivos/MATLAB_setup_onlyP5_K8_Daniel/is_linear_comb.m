%% Copyright (C) 2019 MateusMarcuzzo
%% 
%% This program is free software: you can redistribute it and/or modify it
%% under the terms of the GNU General Public License as published by
%% the Free Software Foundation, either version 3 of the License, or
%% (at your option) any later version.
%% 
%% This program is distributed in the hope that it will be useful, but
%% WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%% GNU General Public License for more details.
%% 
%% You should have received a copy of the GNU General Public License
%% along with this program.  If not, see
%% <https://www.gnu.org/licenses/>.

%% -*- texinfo -*- 
%% @deftypefn {} {@var{retval} =} is_linear_comb (@var{input1}, @var{input2})
%%
%% @seealso{}
%% @end deftypefn

%% Author: MateusMarcuzzo <MateusMarcuzzo@PC-MATEUS>
%% Created: 2019-04-22


% This function is not being used, but it's somehow used in GLICA_function (10/05/2019)
% It was a test setup 
function value = is_linear_comb(most_recent_W,row_candidate,P)
    k = size(most_recent_W,1);
    value = any(sum(abs(mod(generate_pai_P(k,P)(1:end-1,:) * most_recent_W,P) - row_candidate),2)
    ==0);
    
%    mod(mod(generate_pai_P(k,P)(1:end-1,:) * most_recent_W,P) - row_candidate,P)
%    mod(generate_pai_P(k,P)(1:end-1,:) * most_recent_W,P)
    
end
