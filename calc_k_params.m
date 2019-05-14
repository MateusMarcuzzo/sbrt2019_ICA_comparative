function [slopes, consts,range]=calc_k_params(k)
    range_length=0.5/k;
    range=[0:range_length:0.5]';
    mid_range_points=[range_length/2:range_length:0.5-range_length/2]';
    slopes=log2(1-mid_range_points)-log2(mid_range_points);
    ent_at_mid_range_points=-mid_range_points.*log2(mid_range_points)-(1-mid_range_points).*log2(1-mid_range_points);
    consts=ent_at_mid_range_points-slopes.*mid_range_points;
    
    %slopes=[0.8; 0.7; 0.6;];
    %consts=[0.2; 0.4; 0.5;];



end
